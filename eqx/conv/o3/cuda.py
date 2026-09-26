"""Resource-aware sparse contractions with bounded radial matrix products."""

from collections import defaultdict, deque
from functools import lru_cache, partial

import torch

from ...kernels.cuda import kernels, runtime
from ...utils import parse_metadata
from ..graph import prepare_graph
from ..radial import project
from .codegen import convolution_source, fused_source

CHUNK_SIZE = 65536
WORKSPACE_BYTES = 512 << 20
ROW_SIZE = 8
REGISTER_LIMIT = 160


@lru_cache(maxsize=256)
def execution_plan(
    metadata, program, dimensions, shared, dtype, grouped, device, initialize
):
    """Group shared inputs and derivative factors using compiled resource usage."""
    paths, _, _, *geometry = parse_metadata(metadata)
    outputs = 1 + max(slot for _, _, pairs in program for _, slot in pairs)
    group_attrs = any(role == 3 for _, _, pairs in program for role, _ in pairs)
    groups = defaultdict(list)
    for path in paths:
        groups[path[3], path[1 if group_attrs else 0]].append(path)
    pending = deque(
        (tuple(sorted(entries, key=lambda p: (p[1], p[2]))), program)
        for entries in groups.values()
    )
    accepted = []
    scheduled = []
    while pending:
        candidates = []
        while pending:
            paths, terms = pending.popleft()
            paths = tuple(
                path
                for path in paths
                if path[8] >= 0
                or any(
                    not weighted and any(role != 1 for role, _ in pairs)
                    for _, weighted, pairs in terms
                )
            )
            if not paths:
                continue
            roles = {role for _, _, pairs in terms for role, _ in pairs}
            nodes = roles & {0, 4}
            owner = (1 if 4 in nodes else 0 if 0 in nodes else -1) if grouped else -1
            code = convolution_source(
                paths,
                terms,
                dimensions,
                shared,
                dtype,
                owner,
                outputs,
                initialize,
                normalization=geometry[0] if geometry else None,
                use_generators=geometry[1] if len(geometry) > 1 else False,
                angular_derivatives=geometry[2] if len(geometry) > 2 else False,
                direction_offset=geometry[3] if len(geometry) > 3 else 0,
            )
            candidates.append((code, paths, terms, owner))
        compiled = kernels([code for code, *_ in candidates], device)
        for code, paths, terms, owner in candidates:
            kernel = compiled[code]
            if kernel.local_bytes or kernel.registers > REGISTER_LIMIT:
                if len(terms) > 1:
                    half = len(terms) // 2
                    pending.extend(((paths, terms[:half]), (paths, terms[half:])))
                    continue
                if len(paths) > 1:
                    boundaries = [
                        i
                        for i in range(1, len(paths))
                        if paths[i - 1][1] != paths[i][1]
                    ]
                    half = min(
                        boundaries or range(1, len(paths)),
                        key=lambda i: abs(2 * i - len(paths)),
                    )
                    pending.extend(((paths[:half], terms), (paths[half:], terms)))
                    continue
            threads = max(
                (64, 128, 256),
                key=lambda n: (n * kernel.active_blocks(n, 0), -abs(n - 128)),
            )
            accepted.append((kernel, paths[0][3], owner, threads))
            scheduled.append((paths, terms, owner))
    separate, accepted = accepted, []
    pending = deque([tuple(range(len(scheduled)))]) if scheduled else deque()
    while pending:
        candidates = []
        while pending:
            indices = pending.popleft()
            if len(indices) == 1:
                accepted.append(separate[indices[0]])
                continue
            phases = tuple(scheduled[i] for i in indices)
            code = fused_source(
                phases,
                dimensions,
                shared,
                dtype,
                outputs,
                initialize,
                normalization=geometry[0] if geometry else None,
                use_generators=geometry[1] if len(geometry) > 1 else False,
                angular_derivatives=geometry[2] if len(geometry) > 2 else False,
                direction_offset=geometry[3] if len(geometry) > 3 else 0,
            )
            candidates.append((code, indices, phases))
        compiled = kernels([code for code, _, _ in candidates], device)
        for code, indices, phases in candidates:
            kernel = compiled[code]
            if kernel.local_bytes or kernel.registers > 192:
                half = len(indices) // 2
                pending.extend((indices[:half], indices[half:]))
                continue
            threads = max(
                (64, 128, 256),
                key=lambda n: (n * kernel.active_blocks(n, 0), -abs(n - 128)),
            )
            accepted.append(
                (
                    kernel,
                    max(paths[0][3] for paths, _, _ in phases),
                    tuple(owner for _, _, owner in phases),
                    threads,
                )
            )
    return tuple(accepted)


def contract_direct(metadata, source, target, calls, shared=None, initialize=()):
    """Share angular factors across derivative terms and reduce on their owners."""
    operands, results = [], []
    operand_ids, result_ids, program = {}, {}, []
    shared_values = {}
    for outputs, values, destinations, weighted in calls:
        mapping = []
        for i, value in enumerate(values):
            if id(value) not in operand_ids:
                operand_ids[id(value)] = len(operands)
                operands.append(value)
            pointer = operand_ids[id(value)]
            mapping.append(pointer)
            if shared is not None and i in (1, 3):
                shared_values[pointer] = shared[0 if i == 1 else 1]
        pairs = []
        for role, value in zip(outputs, destinations):
            if id(value) not in result_ids:
                result_ids[id(value)] = len(results)
                results.append(value)
            pairs.append((role, result_ids[id(value)]))
        program.append((tuple(mapping), weighted, tuple(pairs)))
    phases = execution_plan(
        metadata,
        tuple(program),
        tuple(value.size(1) for value in operands),
        tuple(
            shared_values.get(i, value.size(0) == 1) for i, value in enumerate(operands)
        ),
        "float" if operands[0].dtype == torch.float32 else "double",
        source.numel() >= 1024,
        operands[0].device,
        tuple(result_ids[id(value)] for value in initialize),
    )
    orders = {
        owner: prepare_graph(source, target, owner)
        for owner in {
            owner
            for phase in phases
            for owner in (phase[2] if isinstance(phase[2], tuple) else (phase[2],))
        }
        if owner >= 0
    }
    pointers = [value.data_ptr() for value in (*operands, *results)]
    launches = []
    for kernel, width, owner, threads in phases:
        if isinstance(owner, tuple):
            arguments = [
                *pointers,
                source.data_ptr(),
                target.data_ptr(),
                orders[0].data_ptr() if 0 in orders else 0,
                orders[1].data_ptr() if 1 in orders else 0,
                source.numel(),
                ROW_SIZE,
            ]
            warps = threads // 32
            edge_blocks = (source.numel() + warps - 1) // warps
            node_blocks = (source.numel() + ROW_SIZE * warps - 1) // (ROW_SIZE * warps)
            count = sum(node_blocks if index >= 0 else edge_blocks for index in owner)
            launches.append((kernel, arguments, count, (width + 31) // 32, threads, 0))
            continue
        rows = ROW_SIZE if owner >= 0 else 1
        count = (source.numel() + rows - 1) // rows
        arguments = [
            *pointers,
            source.data_ptr(),
            target.data_ptr(),
            orders[owner].data_ptr() if owner >= 0 else 0,
            source.numel(),
            count,
            rows,
        ]
        launches.append(
            (
                kernel,
                arguments,
                (count + threads // 32 - 1) // (threads // 32),
                (width + 31) // 32,
                threads,
                0,
            )
        )
    runtime().launch(
        launches, torch.cuda.current_stream(operands[0].device).cuda_stream
    )


def contract(metadata, program, source, target, operands, results):
    """Contract without saving edge messages or full projected edge weights."""
    device, dtype = operands[0].device, operands[0].dtype
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("The O3 CUDA convolution requires float32 or float64.")
    if any(value.device != device or value.dtype != dtype for value in operands):
        raise ValueError("Convolution operands must share a device and dtype.")
    if source.dtype != torch.int64 or target.dtype != torch.int64:
        raise TypeError("Edge indices must have dtype int64.")
    if source.device != device or target.device != device:
        raise ValueError("Edge indices and features must be on the same device.")
    required = {
        mapping[i]
        for mapping, _, pairs in program
        for output, _ in pairs
        for i in range(len(mapping))
        if i != output
    }
    contiguous = {}
    prepared = []
    for i, value in enumerate(operands):
        if i in required and not value.is_contiguous():
            if id(value) not in contiguous:
                contiguous[id(value)] = value.contiguous()
            value = contiguous[id(value)]
        prepared.append(value)
    operands = tuple(prepared)
    source, target = source.contiguous(), target.contiguous()
    calls = [
        (
            tuple(role for role, _ in pairs),
            tuple(operands[i] for i in mapping),
            tuple(results[slot] for _, slot in pairs),
            weighted,
        )
        for mapping, weighted, pairs in program
    ]
    paths, weight_numel, *_ = parse_metadata(metadata)
    projected = [call for call in calls if call[1][2].numel()]
    direct = [call for call in calls if not call[1][2].numel()]
    with torch.cuda.device(device):
        if direct:
            # A zero-sized projection has an identically zero derivative.
            direct = [
                (
                    tuple(role for role in outputs if role != 2),
                    values,
                    tuple(
                        value for role, value in zip(outputs, destinations) if role != 2
                    ),
                    weighted,
                )
                for outputs, values, destinations, weighted in direct
                if any(role != 2 for role in outputs)
            ]
            if direct:
                contract_direct(metadata, source, target, direct)
        if projected:
            project(
                weight_numel,
                source,
                target,
                projected,
                partial(contract_direct, metadata),
                complete=sum(p[3] * p[4] for p in paths if p[8] >= 0) == weight_numel,
                output_role=4,
                chunk_size=CHUNK_SIZE,
                workspace_bytes=WORKSPACE_BYTES,
            )
