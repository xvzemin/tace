"""O(2)-aligned O(3) contraction schedules and bounded radial workspaces."""

from collections import defaultdict
from functools import partial

from ..radial import project as project_radial

INPUT_TILE = 8


def contraction_groups(path_data):
    """Bound cached input rows while retaining compatible paths in each tile."""
    layouts = {}
    for i, (mode, path) in enumerate(path_data):
        inputs = layouts.setdefault((mode, path[2], path[3]), {})
        inputs.setdefault((path[0], path[4]), []).append(i)
    groups = []
    for inputs in layouts.values():
        tile, width = [], 0
        for (_, dim), indices in inputs.items():
            if tile and width + dim > INPUT_TILE:
                groups.append(tuple(tile))
                tile, width = [], 0
            tile.extend(indices)
            width += dim
        if tile:
            groups.append(tuple(tile))
    return tuple(groups)


def schedule(calls):
    """Intern shared factors and collect adjoints before each rotation."""
    pointers, results = [], []
    pointer_ids, result_ids = {}, {}
    xp, yp, messages, adjoints, q = {}, {}, {}, {}, {}
    gx, gdi, gy, gdo, gr, gs = (defaultdict(list) for _ in range(6))
    for outputs, operands, destinations, weighted in calls:
        ids = []
        for value in operands:
            if id(value) not in pointer_ids:
                pointer_ids[id(value)] = len(pointers)
                pointers.append(value)
            ids.append(pointer_ids[id(value)])
        output_ids = {}
        for role, value in zip(outputs, destinations):
            if id(value) not in result_ids:
                result_ids[id(value)] = len(results)
                results.append(value)
            output_ids[role] = result_ids[id(value)]
        x, w, _, di, do, s, y = ids
        if any(i in outputs for i in (1, 4, 5, 6)):
            xi = xp.setdefault((x, di), len(xp))
        if any(i in outputs for i in (0, 1, 3, 5)):
            yi = yp.setdefault((y, do), len(yp))
        if 6 in outputs or 4 in outputs:
            mi = messages.setdefault((xi, w, s, weighted), len(messages))
            if 6 in outputs:
                gy[do, output_ids[6]].append(mi)
            if 4 in outputs:
                gdo[y, output_ids[4]].append(mi)
        if 0 in outputs or 3 in outputs:
            ai = adjoints.setdefault((yi, w, s, weighted), len(adjoints))
            if 0 in outputs:
                gx[di, output_ids[0]].append(ai)
            if 3 in outputs:
                gdi[x, output_ids[3]].append(ai)
        if 1 in outputs or 5 in outputs:
            qi = q.setdefault((xi, yi), len(q))
            if 1 in outputs:
                gr[output_ids[1]].append((qi, s))
            if 5 in outputs:
                gs[output_ids[5]].append((qi, w, weighted))
    accumulators = {}
    input_gradients = []
    for group in (gx, gdi):
        entries = []
        for key, value in group.items():
            terms = tuple(sorted(value))
            accumulator = accumulators.setdefault(terms, len(accumulators))
            entries.append((*key, accumulator))
        input_gradients.append(tuple(entries))
    return (
        tuple(pointers),
        tuple(results),
        tuple(xp),
        tuple(yp),
        tuple(messages),
        tuple(adjoints),
        tuple(q),
        tuple(accumulators),
        *input_gradients,
        *(tuple((*key, *value) for key, value in group.items()) for group in (gy, gdo)),
        *(
            tuple(
                (key, *(entry for term in value for entry in term))
                for key, value in group.items()
            )
            for group in (gr, gs)
        ),
    )


def register_estimate(paths, operations, itemsize):
    """Estimate live 32-bit words per channel, including rotation temporaries."""
    xp, yp, messages, _, q, accumulators, *_ = operations
    inputs, outputs = rotation_groups(paths)
    dim = sum(paths[group[0]][0][4] for group in inputs)
    dim_out = max(sum(len(paths[i][1]) for i in group) for group in outputs)
    live = dim * (len(xp) + len(accumulators))
    live += dim_out * (len(yp) + len(messages))
    live += 2 * max(max(path[4:6]) for path, _ in paths) + len(q) + 8
    return 24 + live * (itemsize // 4)


def rotation_groups(paths):
    """Group input blocks and tile compatible output paths by angular width."""
    inputs, outputs = {}, {}
    for i, (path, _) in enumerate(paths):
        inputs.setdefault((path[0], path[2], path[4], path[6]), []).append(i)
        outputs.setdefault((path[3], path[5], path[7]), {}).setdefault(
            path[1], []
        ).append(i)
    output_tiles = []
    for entries in outputs.values():
        first = next(iter(entries.values()))[0]
        width = max(2, 32 // paths[first][0][5])
        tile = []
        for group in entries.values():
            if tile and len(tile) + len(group) > width:
                output_tiles.append(tuple(tile))
                tile = []
            # Instructions with the same actual destination are summed before
            # the output rotation. Independent irrep entries remain separate.
            tile.extend(group)
        if tile:
            output_tiles.append(tuple(tile))
    return tuple(map(tuple, inputs.values())), tuple(output_tiles)


def split_program(calls):
    """Partition derivative outputs while retaining shared expensive factors."""
    terms = [
        (*prefix, (role,), values, (destination,), weighted)
        for *prefix, outputs, values, destinations, weighted in calls
        for role, destination in zip(outputs, destinations)
    ]
    dependencies = []
    for *prefix, (role,), values, _, weighted in terms:
        x, w, _, di, do, s, y, *vectors = map(id, values)
        factors = set()
        if role in (1, 4, 5, 6) or role >= 7:
            factors.add(("input rotation", x, di))
        if role in (0, 1, 3, 5) or role >= 7:
            factors.add(("output rotation", y, do))
        if role in (0, 3, 4, 5, 6) or role >= 7:
            factors.add(("weight", w, weighted))
        if role in (0, 1, 3, 4, 6) or role >= 7:
            factors.add(("amplitude", s))
        factors.update(
            ("vector rotation", value, di)
            for axis, value in enumerate(vectors)
            if role != 7 + axis
        )
        dependencies.append(factors)
    first = max(range(len(terms)), key=lambda i: len(dependencies[i]))
    second = min(
        (i for i in range(len(terms)) if i != first),
        key=lambda i: (
            len(dependencies[i] & dependencies[first])
            / max(1, len(dependencies[i] | dependencies[first])),
            -len(dependencies[i]),
        ),
    )
    groups = [[first], [second]]
    factors = [dependencies[first].copy(), dependencies[second].copy()]
    remaining = sorted(
        (i for i in range(len(terms)) if i not in (first, second)),
        key=lambda i: -len(dependencies[i]),
    )
    limit = max(1, (2 * len(terms) + 2) // 3)
    for index in remaining:
        group = min(
            (g for g in range(2) if len(groups[g]) < limit),
            key=lambda g: (
                len(dependencies[index] - factors[g]),
                len(groups[g]),
            ),
        )
        groups[group].append(index)
        factors[group].update(dependencies[index])
    return tuple([terms[i] for i in sorted(group)] for group in groups)


def project(
    plan, source, target, calls, contract, chunk_size=16384, workspace_bytes=512 << 20
):
    """Evaluate shared radial projections in bounded workspaces."""
    return project_radial(
        plan.weight_numel,
        source,
        target,
        calls,
        partial(contract, plan),
        complete=sum(path[2] for _, path in plan.path_data if path[8] >= 0)
        == plan.weight_numel,
        output_role=6,
        chunk_size=chunk_size,
        workspace_bytes=workspace_bytes,
    )
