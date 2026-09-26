"""CUDA execution for O(2)-aligned O(3) tensor-product convolutions."""

from collections import deque
from functools import lru_cache

import torch

from ...kernels.cuda import kernels, launch_config, runtime
from ..graph import prepare_graph
from .codegen import convolution_source
from .schedule import (
    contraction_groups,
    project,
    register_estimate,
    rotation_groups,
    schedule,
    split_program,
)

WORKSPACE_BYTES = 1024 << 20
CHUNK_SIZE = 65536
ROW_SIZE = 4
THREADS = 128


@lru_cache(maxsize=256)
def execution_plan(
    plan, specification, dimensions, shared, dtype, grouped, device, initialize=()
):
    """Cache tensor-free schedules, using compiled resource usage to bound tiles."""
    count = 1 + max(
        i for _, operands, results, _ in specification for i in (*operands, *results)
    )
    tokens = tuple(object() for _ in range(count))
    locations = {id(token): i for i, token in enumerate(tokens)}
    calls = tuple(
        (
            outputs,
            tuple(tokens[i] for i in operands),
            tuple(tokens[i] for i in results),
            weighted,
        )
        for outputs, operands, results, weighted in specification
    )
    phases = []
    for indices in contraction_groups(plan.path_data):
        entries = [(*plan.path_data[i], plan.sparse_paths[i], i) for i in indices]
        entries = [
            (mode, (*path[:6], -1, -1, *path[8:]), cg, i)
            if i in plan.scalar_paths
            else (mode, path, cg, i)
            for mode, path, cg, i in entries
        ]
        entries.sort(key=lambda entry: (entry[1][7], entry[1][0]))
        paths = tuple((mode, path, cg) for mode, path, cg, _ in entries)
        pending = deque([(paths, calls)])
        while pending:
            paths, terms = pending.popleft()
            inputs, results, *operations = schedule(terms)
            if (
                register_estimate(
                    [(path, cg) for _, path, cg in paths],
                    operations,
                    4 if dtype == "float" else 8,
                )
                > 256
                and sum(len(term[0]) for term in terms) > 1
            ):
                pending.extend((paths, group) for group in split_program(terms))
                continue
            roles = {role for outputs, *_ in terms for role in outputs}
            owner = (
                -1 if not grouped else 1 if roles == {6} else 0 if roles == {0} else -1
            )
            # Bound receiver accumulators by actual output entries. Tile the
            # outputs instead of abandoning node ownership for wide products.
            if (
                owner == 1
                and sum(dim for _, dim in {(path[1], path[5]) for _, path, _ in paths})
                * len(operations[8])
                > 64
            ):
                _, groups = rotation_groups([(path, cg) for _, path, cg in paths])
                if len(groups) > 1:
                    tile, width = [], 0
                    for group in groups:
                        extra = sum(
                            dim
                            for _, dim in {
                                (paths[i][1][1], paths[i][1][5]) for i in group
                            }
                        ) * len(operations[8])
                        if tile and width + extra > 64:
                            pending.append((tuple(tile), terms))
                            tile, width = [], 0
                        tile.extend(paths[i] for i in group)
                        width += extra
                    pending.append((tuple(tile), terms))
                    continue
            code, width, shared_bytes = convolution_source(
                paths,
                tuple(operations),
                dimensions,
                shared,
                len(inputs),
                len(results),
                dtype,
                owner,
                tuple(
                    i
                    for i, value in enumerate(results)
                    if locations[id(value)] in initialize
                ),
            )
            phases.append(
                (code, width, shared_bytes, owner, inputs, results, paths, terms)
            )
    # Compile candidates together, then refine only kernels with excessive live
    # state. Splitting is independent of angular degree and never changes paths.
    accepted = []
    while phases:
        compiled = kernels([phase[0] for phase in phases], device)
        pending = []
        for code, width, shared_bytes, owner, inputs, results, paths, terms in phases:
            kernel = compiled[code]
            if kernel.registers > 160 or kernel.local_bytes:
                boundaries = [
                    i
                    for i in range(1, len(paths))
                    if paths[i - 1][1][7] != paths[i][1][7]
                ]
                if sum(len(term[0]) for term in terms) > 1:
                    parts = tuple((paths, group) for group in split_program(terms))
                elif boundaries:
                    midpoint = min(
                        boundaries,
                        key=lambda i: abs(2 * i - len(paths)),
                    )
                    parts = ((paths[:midpoint], terms), (paths[midpoint:], terms))
                elif len(paths) > 1:
                    midpoint = len(paths) // 2
                    parts = ((paths[:midpoint], terms), (paths[midpoint:], terms))
                else:
                    parts = ()
                if parts:
                    for subset, group in parts:
                        operands, destinations, *operations = schedule(group)
                        source, width, shared_bytes = convolution_source(
                            subset,
                            tuple(operations),
                            dimensions,
                            shared,
                            len(operands),
                            len(destinations),
                            dtype,
                            owner,
                            tuple(
                                i
                                for i, value in enumerate(destinations)
                                if locations[id(value)] in initialize
                            ),
                        )
                        pending.append(
                            (
                                source,
                                width,
                                shared_bytes,
                                owner,
                                operands,
                                destinations,
                                subset,
                                group,
                            )
                        )
                    continue
            # Use compiled occupancy to size channel and edge tiles. Wide
            # channels share matrix storage across warps; no degree thresholds.
            common, per_warp = shared_bytes
            sizes = (
                tuple(
                    n for n in (64, 128, 256) if n <= max(64, (width + 31) // 32 * 32)
                )
                if width > 32
                else (64, 128, 256)
            )
            threads = max(
                (n for n in sizes if common + per_warp * (n // 32) <= 49152),
                key=lambda n: (
                    n * kernel.active_blocks(n, common + per_warp * (n // 32)),
                    -abs(n - THREADS),
                ),
            )
            accepted.append(
                (
                    kernel,
                    width,
                    (common, per_warp),
                    threads,
                    owner,
                    tuple(locations[id(value)] for value in inputs),
                    tuple(locations[id(value)] for value in results),
                )
            )
        phases = pending
    return tuple(accepted)


def contract(plan, source, target, calls, shared=None, initialize=()):
    """Execute a cached mixed-adjoint program without edge message storage."""
    x, radial, _, din, dout, amplitudes, y = calls[0][1]
    if shared is None:
        shared = tuple(value.size(0) == 1 for value in (radial, din, dout, amplitudes))
    dimensions = (x.size(1), y.size(1), din.size(1), radial.size(1), amplitudes.size(1))
    values, indices = [], {}
    specification = []
    for outputs, operands, results, weighted in calls:
        slots = []
        for value in (*operands, *results):
            if id(value) not in indices:
                indices[id(value)] = len(values)
                values.append(value)
            slots.append(indices[id(value)])
        specification.append((outputs, tuple(slots[:7]), tuple(slots[7:]), weighted))
    phases = execution_plan(
        plan,
        tuple(specification),
        dimensions,
        shared,
        "float" if x.dtype == torch.float32 else "double",
        source.numel() >= 1024,
        x.device,
        tuple(indices[id(value)] for value in initialize),
    )
    stream = torch.cuda.current_stream(x.device).cuda_stream
    pointers = tuple(value.data_ptr() for value in values)
    edge_pointers = source.data_ptr(), target.data_ptr()
    graphs = {}
    launches = []
    for kernel, width, storage, threads, owner, inputs, results in phases:
        if owner >= 0:
            if owner not in graphs:
                graphs[owner] = prepare_graph(source, target, owner)
            order = graphs[owner]
            count = (source.numel() + ROW_SIZE - 1) // ROW_SIZE
        else:
            order, count = source, source.numel()
        tensors = (
            *(values[i] for i in (*inputs, *results)),
            source,
            target,
            order,
        )
        scalars = (source.numel(), count, ROW_SIZE)
        threads, shared_bytes = launch_config(
            kernel,
            width,
            count,
            storage,
            threads,
            tensors,
            scalars,
            range(len(inputs), len(inputs) + len(results)),
        )
        args = [pointers[i] for i in (*inputs, *results)]
        args.extend((*edge_pointers, order.data_ptr(), *scalars))
        tasks_per_block = 1 if width > 32 else threads // 32
        channels_per_block = threads if width > 32 else 32
        launches.append(
            (
                kernel,
                args,
                (count + tasks_per_block - 1) // tasks_per_block,
                (width + channels_per_block - 1) // channels_per_block,
                threads,
                shared_bytes,
            )
        )
    runtime().launch(launches, stream)


def contract_many(plan, source, target, calls):
    """Consume projected radial weights and their adjoints in bounded chunks."""
    contiguous = {}
    prepared = []
    for outputs, operands, results, weighted in calls:
        if operands[0].dtype not in (torch.float32, torch.float64):
            raise TypeError("EQX CUDA convolutions support float32 and float64.")
        values = []
        for role, value in enumerate(operands):
            if outputs != (role,) and not value.is_contiguous():
                if id(value) not in contiguous:
                    contiguous[id(value)] = value.contiguous()
                value = contiguous[id(value)]
            values.append(value)
        prepared.append((outputs, tuple(values), results, weighted))
    source, target = source.contiguous(), target.contiguous()
    if prepared[0][1][2].numel():
        project(plan, source, target, prepared, contract, CHUNK_SIZE, WORKSPACE_BYTES)
    else:
        contract(plan, source, target, prepared)


@lru_cache(maxsize=256)
def direction_plan(
    metadata, specification, layouts, dtype, grouped, device, initialize=()
):
    """Compile shared tiles of mixed angular and vector adjoints."""
    from .convolution import kernel_plan
    from .direction_codegen import direction_source

    plan = kernel_plan(metadata)
    pending = [
        (
            tuple(
                sorted(
                    group,
                    key=lambda i: (plan.path_data[i][1][7], plan.path_data[i][1][1]),
                )
            ),
            specification,
        )
        for group in contraction_groups(plan.path_data)
    ]
    accepted = []
    while pending:
        sources = [
            direction_source(
                metadata, group, terms, layouts, dtype, initialize=initialize
            )
            for group, terms in pending
        ]
        compiled = kernels([entry[0] for entry in sources], device)
        remaining = []
        for (group, terms), (code, width, storage) in zip(pending, sources):
            kernel = compiled[code]
            if kernel.local_bytes or kernel.registers > 160:
                if len(group) > 1:
                    # Keep the different adjoints together while they share
                    # input/output rotations. Split independent output degrees
                    # before partitioning the derivative program itself.
                    boundaries = [
                        i
                        for i in range(1, len(group))
                        if plan.path_data[group[i - 1]][1][7]
                        != plan.path_data[group[i]][1][7]
                    ]
                    middle = (
                        min(boundaries, key=lambda i: abs(2 * i - len(group)))
                        if boundaries
                        else len(group) // 2
                    )
                    remaining.extend(((group[:middle], terms), (group[middle:], terms)))
                    continue
                if sum(len(term[1]) for term in terms) > 1:
                    remaining.extend(
                        (group, tuple(part)) for part in split_program(terms)
                    )
                    continue
            accepted.append((kernel, width, storage, group, terms))
        pending = remaining
    # Try ownership on already bounded tiles. Do not split shared rotations
    # merely to accommodate the additional persistent node accumulators.
    candidates = []
    for _, _, _, group, terms in accepted:
        roles = {role for _, outputs, *_ in terms for role in outputs}
        owner = -1 if not grouped else 0 if 0 in roles else 1 if 6 in roles else -1
        code = (
            direction_source(metadata, group, terms, layouts, dtype, owner, initialize)[
                0
            ]
            if owner >= 0
            else None
        )
        candidates.append((code, owner))
    compiled = kernels([code for code, _ in candidates if code is not None], device)
    phases = []
    for (kernel, width, storage, _, _), (code, owner) in zip(accepted, candidates):
        choices = [(kernel, -1)]
        if (
            code is not None
            and not compiled[code].local_bytes
            and compiled[code].registers <= 160
        ):
            choices.append((compiled[code], owner))
        sizes = tuple(
            n
            for n in (64, 128, 256)
            if (width <= 32 or n <= max(64, (width + 31) // 32 * 32))
            and (storage if width > 32 else storage * (n // 32)) <= 49152
        )
        # Reduction savings must not come at the cost of fewer resident
        # warps. Prefer ownership when both variants admit the same occupancy.
        kernel, owner, threads = max(
            ((candidate, axis, n) for candidate, axis in choices for n in sizes),
            key=lambda choice: (
                choice[2]
                * choice[0].active_blocks(
                    choice[2], storage if width > 32 else storage * (choice[2] // 32)
                ),
                choice[1] >= 0,
                -abs(choice[2] - THREADS),
            ),
        )
        phases.append(
            (
                kernel,
                width,
                threads,
                (storage, 0) if width > 32 else (0, storage),
                owner,
            )
        )
    return tuple(phases)


def contract_directions(metadata, source, target, calls):
    """Stream radial projections once across all geometric derivative terms."""
    from .convolution import kernel_plan

    plan = kernel_plan(metadata)
    contiguous, prepared = {}, []
    for rank, outputs, operands, results, weighted in calls:
        values = []
        for role, value in enumerate(operands):
            if outputs != (role,) and not value.is_contiguous():
                if id(value) not in contiguous:
                    contiguous[id(value)] = value.contiguous()
                value = contiguous[id(value)]
            values.append(value)
        prepared.append((rank, outputs, tuple(values), results, weighted))

    def execute(plan, source, target, terms, shared=None, initialize=()):
        direct_metadata, remaining, remaining_metadata = harmonic_plan(metadata)
        if direct_metadata is not None:
            from ..o3.cuda import contract_direct

            direct = []
            roles = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}
            for _, outputs, operands, results, weighted in terms:
                x, radial, projection, frame, _, amplitudes, y, *vectors = operands
                values = (
                    x,
                    radial,
                    projection,
                    amplitudes,
                    y,
                    frame,
                    *vectors,
                )
                direct.append(
                    (
                        tuple(roles[r] if r < 7 else r - 1 for r in outputs),
                        values,
                        results,
                        weighted,
                    )
                )
            contract_direct(
                direct_metadata,
                source,
                target,
                direct,
                None if shared is None else (shared[0], shared[3]),
                initialize,
            )
            plan = remaining
            if not plan.path_data:
                return
        if all(rank == 0 for rank, *_ in terms):
            return contract(
                plan, source, target, [term[1:] for term in terms], shared, initialize
            )
        values, locations, specification = [], {}, []
        for rank, outputs, operands, results, weighted in terms:
            slots = []
            for value in (*operands, *results):
                if id(value) not in locations:
                    locations[id(value)] = len(values)
                    values.append(value)
                slots.append(locations[id(value)])
            specification.append(
                (
                    rank,
                    outputs,
                    tuple(slots[: len(operands)]),
                    tuple(slots[len(operands) :]),
                    weighted,
                )
            )
        layouts = tuple((value.size(0) == 1, *value.stride()) for value in values)
        phases = direction_plan(
            remaining_metadata,
            tuple(specification),
            layouts,
            "float" if values[0].dtype == torch.float32 else "double",
            source.numel() >= 1024,
            values[0].device,
            tuple(locations[id(value)] for value in initialize),
        )
        pointers = [value.data_ptr() for value in (*values, source, target)]
        destinations = {slot for _, _, _, slots, _ in specification for slot in slots}
        graphs = {}
        launches = []
        for kernel, width, threads, storage, owner in phases:
            if owner not in graphs:
                if owner >= 0:
                    order = prepare_graph(source, target, owner)
                    count = (source.numel() + ROW_SIZE - 1) // ROW_SIZE
                else:
                    order, count = source, source.numel()
                tensors = (*values, source, target, order)
                args = [
                    *pointers,
                    order.data_ptr(),
                    source.numel(),
                    count,
                    ROW_SIZE,
                ]
                graphs[owner] = tensors, args, count
            tensors, args, count = graphs[owner]
            threads, shared_bytes = launch_config(
                kernel,
                width,
                count,
                storage,
                threads,
                tensors,
                (source.numel(), count, ROW_SIZE),
                destinations,
            )
            tasks = 1 if width > 32 else threads // 32
            channels = threads if width > 32 else 32
            launches.append(
                (
                    kernel,
                    args,
                    (count + tasks - 1) // tasks,
                    (width + channels - 1) // channels,
                    threads,
                    shared_bytes,
                )
            )
        runtime().launch(
            launches, torch.cuda.current_stream(values[0].device).cuda_stream
        )

    source, target = source.contiguous(), target.contiguous()
    if prepared[0][2][2].numel():
        project(plan, source, target, prepared, execute, CHUNK_SIZE, WORKSPACE_BYTES)
    else:
        execute(plan, source, target, prepared)


@lru_cache(maxsize=256)
def harmonic_plan(metadata):
    """Select low-degree harmonic paths for direct sparse contractions."""
    from dataclasses import replace

    from e3nn import o3

    from .convolution import kernel_plan

    plan = kernel_plan(metadata)
    if not plan.harmonic_degrees:
        return None, plan, metadata
    paths, keep = [], []
    for i, ((_, path), degree, entries) in enumerate(
        zip(plan.path_data, plan.harmonic_degrees, plan.sparse_paths)
    ):
        start, end, mul, _, dim, dim_out, _, _, weight, harmonic = path
        if degree > 2:
            keep.append(i)
            continue
        cg = o3.wigner_3j(
            (dim - 1) // 2,
            degree,
            (dim_out - 1) // 2,
            dtype=torch.float64,
            device="cpu",
        )
        scale = sum(c * float(cg[a, degree, b]) for a, b, c in entries)
        scale /= float(cg[:, degree, :].square().sum()) * (2 * degree + 1) ** 0.5
        paths.append(
            (
                start,
                harmonic,
                end,
                mul,
                1,
                dim,
                2 * degree + 1,
                dim_out,
                weight,
                scale,
                tuple((*row, float(cg[tuple(row)])) for row in cg.nonzero().tolist()),
            )
        )
    if not paths:
        return None, plan, metadata
    remainder = replace(
        plan,
        path_data=tuple(plan.path_data[i] for i in keep),
        sparse_paths=tuple(plan.sparse_paths[i] for i in keep),
        scalar_paths=tuple(j for j, i in enumerate(keep) if i in plan.scalar_paths),
        harmonic_degrees=tuple(plan.harmonic_degrees[i] for i in keep),
    )
    return (
        repr(
            (
                tuple(paths),
                plan.weight_numel,
                plan.has_unweighted,
                "component",
                True,
                True,
                4,
            )
        ),
        remainder,
        repr(
            (
                remainder.path_data,
                remainder.weight_numel,
                remainder.has_unweighted,
                remainder.sparse_paths,
                remainder.scalar_paths,
                remainder.harmonic_degrees,
            )
        ),
    )
