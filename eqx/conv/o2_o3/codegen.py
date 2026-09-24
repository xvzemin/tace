"""CUDA source generation for aligned O(3) tensor-product contractions."""

import math
from functools import lru_cache

from ...kernels.codegen import HEADER
from .schedule import rotation_groups


@lru_cache(maxsize=512)
def convolution_source(
    paths, operations, dimensions, shared, inputs, outputs, dtype, owner, initialize=()
):
    """Emit shared input rotations and path/channel tiles of output rotations."""
    xp, yp, messages, adjoints, qs, accumulators, gx, gdi, gy, gdo, gr, gs = operations
    xdim, ydim, ddim, rdim, sdim = dimensions
    mul = paths[0][1][2]
    block_channels = mul > 32
    barrier = "__syncthreads();" if block_channels else "__syncwarp();"
    input_groups, output_groups = rotation_groups([(path, cg) for _, path, cg in paths])
    input_paths = [paths[group[0]][1] for group in input_groups]
    input_index = {j: i for i, group in enumerate(input_groups) for j in group}
    support = [{b: (a, c) for a, b, c in cg} for _, _, cg in paths]
    destinations = {}
    for j, (_, path, _) in enumerate(paths):
        destinations.setdefault((path[1], path[3], path[5], path[7]), []).append(j)
    merged = {j for group in destinations.values() if len(group) > 1 for j in group}
    # Wide channel tiles share one matrix across the block's warps. Narrow
    # tiles keep independent edges in each warp without block synchronization.
    matrices = {}
    for path in input_paths:
        if path[6] < 0:
            continue
        for pd in (*[pd for _, pd in xp], *[pd for pd, *_ in gx]):
            matrices[pd, "ei", path[6], path[4]] = None
    for group in destinations.values():
        path = paths[group[0]][1]
        if len(group) > 1 and path[7] >= 0:
            for pd, *_ in gy:
                matrices[pd, "eo", path[7], path[5]] = None
    # C has one nonzero per supported order. Fold it into the output Wigner
    # rows once per tile, then reuse those rows across channels and transposes.
    coupled_offsets, coupled_entries = {}, {}
    coupled_size = 0
    for j, (_, path, _) in enumerate(paths):
        if path[7] < 0:
            continue
        for pd in (
            *[pd for _, pd in yp],
            *[pd for pd, *_ in gy if j not in merged],
        ):
            for b, (_, c) in support[j].items():
                key = pd, path[7] + b * path[5], path[5], c
                if key not in coupled_entries:
                    coupled_entries[key] = coupled_size
                    coupled_size += path[5]
                coupled_offsets[pd, j, b] = coupled_entries[key]
    gradients = {}
    gradient_size = 0
    for entries, shared_edge, blocks in (
        (gdo, shared[2], [(path[7], path[5]) for _, path, _ in paths]),
        (gdi, shared[1], [(path[6], path[4]) for path in input_paths]),
    ):
        for offset, size in blocks:
            for _, g, *_ in entries:
                key = g, "0" if shared_edge else "edge", offset, size
                if key not in gradients:
                    gradients[key] = gradient_size
                    gradient_size += size**2
    itemsize = 4 if dtype == "float" else 8
    if gradient_size * 4 * itemsize > 16384:
        gradients, gradient_size = {}, 0
    matrix_size = sum(size * size for _, _, _, size in matrices)
    coupled_staging = (coupled_size + gradient_size) * 4 * itemsize <= 24576
    staging = (
        matrix_size + (coupled_size if coupled_staging else 0) + gradient_size
    ) * 4 * itemsize <= 24576
    if staging:
        offset = 0
        for key in matrices:
            matrices[key] = offset
            offset += key[3] ** 2
    args = [f"const T* __restrict__ p{i}" for i in range(inputs)]
    args += [f"T* __restrict__ g{i}" for i in range(outputs)]
    args += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* order",
        "int64_t edges",
        "int64_t task_count",
        "int64_t row_size",
    ]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ",".join(args) + ") {",
        "const int lane = threadIdx.x & 31;",
        (
            "const int64_t item = blockIdx.x;"
            if block_channels
            else "const int64_t item = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;"
        ),
        (
            "const int channel = blockIdx.y * blockDim.x + threadIdx.x;"
            if block_channels
            else "const int channel = blockIdx.y * 32 + lane;"
        ),
        f"const bool active = channel < {mul};",
    ]
    emit = lines.append
    shared_size = (
        (matrix_size if staging else 0)
        + (coupled_size if coupled_staging else 0)
        + gradient_size
    )
    if shared_size:
        emit("extern __shared__ T storage[];")
        emit(
            "T* warp_storage = storage;"
            if block_channels
            else f"T* warp_storage = storage + (threadIdx.x / 32) * {shared_size};"
        )
    offset = 0
    if staging and matrix_size:
        emit("volatile T* matrix = warp_storage;")
        offset += matrix_size
    if coupled_staging and coupled_size:
        emit(f"volatile T* coupled_matrix = warp_storage + {offset};")
        offset += coupled_size
    if gradient_size:
        emit(
            f"T* matrix_gradient = warp_storage + {offset}"
            + (f" + (threadIdx.x / 32) * {gradient_size}" if block_channels else "")
            + ";"
        )

    def load(p, offset):
        return f"(active ? p{p}[{offset}] : T(0))"

    def vector(name, p, node, size, offset, channels, index):
        for a in range(size):
            emit(
                f"T {name}{a} = {load(p, f'{node} * {channels[0]} + {offset + a * channels[1]} + {index}')};"
            )

    def rotated(
        vectors, p, e, offset, size, transpose=False, consume=None, couplings=None
    ):
        # Each Wigner entry is loaded once for the whole path/channel tile.
        # Consume forward rows immediately instead of retaining all outputs.
        for a in range(size):
            selected = [(name, values) for name, values, rows in vectors if a in rows]
            if not selected:
                continue
            if consume is not None:
                emit("{")
            for name, _ in selected:
                emit(f"T {name}{a} = 0;")
            for b in range(size):
                if offset < 0 and a != b:
                    continue
                terms = [
                    (name, values[b])
                    for name, values in selected
                    if values[b] is not None
                ]
                if not terms:
                    continue
                index = b * size + a if transpose else a * size + b
                coefficients = {}
                for name, value in terms:
                    if offset < 0:
                        coefficient = (
                            "T(1)"
                            if couplings is None
                            else f"T({support[couplings[name]][a][1]:.17g})"
                        )
                    elif couplings is None:
                        coefficient = (
                            f"matrix[{matrices[p, e, offset, size] + index}]"
                            if staging
                            else f"p{p}[{e} * {ddim} + {offset + index}]"
                        )
                    else:
                        j, order = couplings[name], b if transpose else a
                        coefficient = (
                            f"coupled_matrix[{coupled_offsets[p, j, order] + (a if transpose else b)}]"
                            if coupled_staging
                            else f"(T({support[j][order][1]:.17g}) * p{p}[{e} * {ddim} + {offset + index}])"
                        )
                    coefficients.setdefault(coefficient, []).append((name, value))
                for coefficient, entries in coefficients.items():
                    emit(f"{{ const T d = {coefficient};")
                    for name, value in entries:
                        emit(f"{name}{a} = fma(d, {value}, {name}{a});")
                    emit("}")
            if consume is not None:
                for name, _ in selected:
                    consume(name, a)
                emit("}")

    def store(g, offset, value, reduced=False, private=False, assign=False):
        if reduced:
            emit(
                f"{{ T reduced_value = warp_sum({value}); if (lane == 0) atomicAdd(g{g} + {offset}, reduced_value); }}"
            )
        elif private:
            emit(f"if (active) g{g}[{offset}] {'=' if assign else '+='} {value};")
        else:
            emit(f"if (active) atomicAdd(g{g} + {offset}, {value});")

    if owner >= 0:
        index = "source" if owner == 0 else "target"
        emit("if (item >= task_count) return;")
        emit("const int64_t stop = min((item + 1) * row_size, edges);")
        emit("int64_t end = item * row_size;")
        emit("while (end < stop) {")
        emit("const int64_t begin = end;")
        emit(f"const int64_t node = {index}[order[begin]];")
        emit(f"do {{ ++end; }} while (end < stop && {index}[order[end]] == node);")
        emit(
            f"const bool exclusive = (begin == 0 || {index}[order[begin - 1]] != node) "
            f"&& (end == edges || {index}[order[end]] != node);"
        )
        if owner == 0:
            for k, path in enumerate(input_paths):
                for i, _ in enumerate(gx):
                    for a in range(path[4]):
                        emit(f"T total_x{k}_{i}_{a} = 0;")
        else:
            for group in destinations.values():
                j = group[0]
                path = paths[j][1]
                for i, _ in enumerate(gy):
                    for a in range(path[5]):
                        emit(f"T total_y{j}_{i}_{a} = 0;")
        emit("for (int64_t position = begin; position < end; ++position) {")
        emit("const int64_t edge = order[position];")
    else:
        emit("if (item >= edges) return;")
        emit("const int64_t edge = item;")
    emit("const int64_t src = source[edge], dst = target[edge];")
    for name, value in zip(("er", "ei", "eo", "es"), shared):
        emit(f"const int64_t {name} = {'0' if value else 'edge'};")
    if gradient_size:
        emit(
            f"for (int k = lane; k < {gradient_size}; k += 32) matrix_gradient[k] = 0;"
        )
    thread_start = "threadIdx.x" if block_channels else "lane"
    thread_stride = "blockDim.x" if block_channels else "32"
    if staging and matrix_size:
        for (pd, e, offset, size), location in matrices.items():
            emit(
                f"for (int k = {thread_start}; k < {size * size}; k += {thread_stride}) "
                f"matrix[{location} + k] = p{pd}[{e} * {ddim} + {offset} + k];"
            )
    if coupled_staging and coupled_size:
        for (pd, offset, size, c), location in coupled_entries.items():
            emit(
                f"for (int k = {thread_start}; k < {size}; k += {thread_stride}) "
                f"coupled_matrix[{location} + k] = T({c:.17g}) * p{pd}[eo * {ddim} + {offset} + k];"
            )
    if (staging and matrix_size) or (coupled_staging and coupled_size) or gradient_size:
        emit(barrier)
    for k, path in enumerate(input_paths):
        start, dim, dstart = path[0], path[4], path[6]
        rows = {a for j in input_groups[k] for a, _, _ in paths[j][2]}
        for i, (px, pd) in enumerate(xp):
            vector(f"x{k}_{i}_", px, "src", dim, start, (xdim, mul), "channel")
            rotated(
                [(f"lx{k}_{i}_", [f"x{k}_{i}_{a}" for a in range(dim)], rows)],
                pd,
                "ei",
                dstart,
                dim,
            )
        for i, _ in enumerate(accumulators):
            for a in range(dim):
                emit(f"T dx{k}_{i}_{a} = 0;")

    for group in output_groups:
        odim, ostart = paths[group[0]][1][5], paths[group[0]][1][7]
        emit("{")
        for i, (py, pd) in enumerate(yp):
            for j in group:
                vector(
                    f"y{j}_{i}_",
                    py,
                    "dst",
                    odim,
                    paths[j][1][1],
                    (ydim, mul),
                    "channel",
                )
            rotated(
                [
                    (f"ly{j}_{i}_", [f"y{j}_{i}_{a}" for a in range(odim)], support[j])
                    for j in group
                ],
                pd,
                "eo",
                ostart,
                odim,
                couplings={f"ly{j}_{i}_": j for j in group},
            )

        def weight_value(j, p, weighted):
            weight = paths[j][1][8]
            if weight < 0:
                return "T(0)" if weighted else "T(1)"
            return load(p, f"er * {rdim} + {weight} + channel")

        def amplitude(j, p):
            return f"p{p}[es * {sdim} + {paths[j][1][9]}]"

        for j in group:
            k = input_index[j]
            for i, (xi, pw, ps, weighted) in enumerate(messages):
                emit(
                    f"T ws{j}_{i} = {weight_value(j, pw, weighted)} * {amplitude(j, ps)};"
                )
                for b, (a, _) in support[j].items():
                    emit(f"T msg{j}_{i}_{b} = lx{k}_{xi}_{a} * ws{j}_{i};")
            for t in sorted({t for _, _, *terms in gdo for t in terms}):
                for b, (_, c) in support[j].items():
                    emit(f"T cmsg{j}_{t}_{b} = T({c:.17g}) * msg{j}_{t}_{b};")

        for i, (pd, g, *terms) in enumerate(gy):
            output_paths = {}
            for j in group:
                output_paths.setdefault(paths[j][1][1], []).append(j)
            names = {
                f"out{indices[0]}_{i}_": indices[0] for indices in output_paths.values()
            }

            def consume(name, a):
                j = names[name]
                if owner == 1:
                    emit(f"total_y{j}_{i}_{a} += {name}{a};")
                else:
                    store(
                        g,
                        f"dst * {ydim} + {paths[j][1][1] + a * mul} + channel",
                        f"{name}{a}",
                    )

            independent = [
                indices[0] for indices in output_paths.values() if len(indices) == 1
            ]
            rotated(
                [
                    (
                        f"out{j}_{i}_",
                        [
                            " + ".join(f"msg{j}_{t}_{a}" for t in terms)
                            if a in support[j]
                            else None
                            for a in range(odim)
                        ],
                        range(odim),
                    )
                    for j in independent
                ],
                pd,
                "eo",
                ostart,
                odim,
                True,
                consume,
                {f"out{j}_{i}_": j for j in independent},
            )
            for indices in output_paths.values():
                if len(indices) == 1:
                    continue
                j = indices[0]
                values = []
                for b in range(odim):
                    contributions = [
                        (index, support[index][b][1], t)
                        for index in indices
                        if b in support[index]
                        for t in terms
                    ]
                    if not contributions:
                        values.append(None)
                        continue
                    name = f"sum{j}_{i}_{b}"
                    emit(f"T {name} = 0;")
                    for index, coefficient, t in contributions:
                        emit(
                            f"{name} = fma(T({coefficient:.17g}), msg{index}_{t}_{b}, {name});"
                        )
                    values.append(name)
                rotated(
                    [(f"out{j}_{i}_", values, range(odim))],
                    pd,
                    "eo",
                    ostart,
                    odim,
                    True,
                    consume,
                )
        for i, (py, g, *terms) in enumerate(gdo):
            for j in group:
                vector(
                    f"oy{j}_{i}_",
                    py,
                    "dst",
                    odim,
                    paths[j][1][1],
                    (ydim, mul),
                    "channel",
                )
            for a in range(odim):
                active_paths = [j for j in group if a in support[j]]
                if not active_paths:
                    continue
                for b in range(odim):
                    emit("{ T value = 0;")
                    for j in active_paths:
                        value = " + ".join(f"cmsg{j}_{t}_{a}" for t in terms)
                        emit(f"value = fma(({value}), oy{j}_{i}_{b}, value);")
                    if gradient_size:
                        location = (
                            gradients[g, "0" if shared[2] else "edge", ostart, odim]
                            + a * odim
                            + b
                        )
                        emit(
                            f"value = warp_sum(value); if (lane == 0) matrix_gradient[{location}] += value;"
                        )
                    else:
                        store(
                            g,
                            f"eo * {ddim} + {ostart + a * odim + b}",
                            "value",
                            True,
                        )
                    emit("}")
        for j in group:
            k = input_index[j]
            weight, harmonic = paths[j][1][8:10]
            emit("{")
            for i, (yi, pw, ps, weighted) in enumerate(adjoints):
                emit(
                    f"T adj{i} = {weight_value(j, pw, weighted)} * {amplitude(j, ps)};"
                )
                for t, terms in enumerate(accumulators):
                    count = terms.count(i)
                    if count:
                        for a, b, _ in paths[j][2]:
                            emit(
                                f"dx{k}_{t}_{a} = fma(T({count}) * adj{i}, ly{j}_{yi}_{b}, dx{k}_{t}_{a});"
                            )
            for i, (xi, yi) in enumerate(qs):
                emit(f"T q{i} = 0;")
                for b, (a, _) in support[j].items():
                    emit(f"q{i} = fma(lx{k}_{xi}_{a}, ly{j}_{yi}_{b}, q{i});")
            if weight >= 0:
                for g, *terms in gr:
                    value = " + ".join(
                        f"q{terms[t]} * {amplitude(j, terms[t + 1])}"
                        for t in range(0, len(terms), 2)
                    )
                    store(
                        g,
                        f"er * {rdim} + {weight} + channel",
                        value,
                        private=not shared[0],
                        assign=g in initialize,
                    )
            for g, *terms in gs:
                value = " + ".join(
                    f"q{terms[t]} * {weight_value(j, terms[t + 1], terms[t + 2])}"
                    for t in range(0, len(terms), 3)
                )
                store(g, f"es * {sdim} + {harmonic}", value, True)
            emit("}")
        emit("}")
    for k, path in enumerate(input_paths):
        start, dim, dstart = path[0], path[4], path[6]
        for i, (pd, g, accumulator) in enumerate(gx):
            rotated(
                [
                    (
                        f"back{k}_{i}_",
                        [f"dx{k}_{accumulator}_{a}" for a in range(dim)],
                        range(dim),
                    )
                ],
                pd,
                "ei",
                dstart,
                dim,
                True,
            )
            for a in range(dim):
                if owner == 0:
                    emit(f"total_x{k}_{i}_{a} += back{k}_{i}_{a};")
                else:
                    store(
                        g,
                        f"src * {xdim} + {start + a * mul} + channel",
                        f"back{k}_{i}_{a}",
                    )
        for i, (px, g, accumulator) in enumerate(gdi):
            vector(f"ox{k}_{i}_", px, "src", dim, start, (xdim, mul), "channel")
            for a in range(dim):
                for b in range(dim):
                    value = f"dx{k}_{accumulator}_{a} * ox{k}_{i}_{b}"
                    if gradient_size:
                        location = (
                            gradients[g, "0" if shared[1] else "edge", dstart, dim]
                            + a * dim
                            + b
                        )
                        emit(
                            f"{{ T value = warp_sum({value}); "
                            f"if (lane == 0) matrix_gradient[{location}] += value; }}"
                        )
                    else:
                        store(g, f"ei * {ddim} + {dstart + a * dim + b}", value, True)
    if gradient_size:
        emit(barrier)
        for (g, edge, offset, size), location in gradients.items():
            emit(
                f"for (int k = {thread_start}; k < {size * size}; k += {thread_stride}) {{"
            )
            if block_channels:
                base = shared_size - gradient_size
                emit(
                    f"T value = 0; for (int w = 0; w < blockDim.x / 32; ++w) "
                    f"value += storage[{base} + w * {gradient_size} + {location} + k];"
                )
            else:
                emit(f"T value = matrix_gradient[{location} + k];")
            emit(f"atomicAdd(g{g} + {edge} * {ddim} + {offset} + k, value); }}")
    if owner >= 0:
        if (staging and matrix_size) or coupled_size or gradient_size:
            emit(barrier)
        emit("}")

        def owned_store(g, offset, value):
            emit(
                f"if (active && begin < end) {{ if (exclusive) g{g}[{offset}] += {value}; else atomicAdd(g{g} + {offset}, {value}); }}"
            )

        if owner == 0:
            for k, path in enumerate(input_paths):
                for i, (_, g, _) in enumerate(gx):
                    for a in range(path[4]):
                        owned_store(
                            g,
                            f"node * {xdim} + {path[0] + a * mul} + channel",
                            f"total_x{k}_{i}_{a}",
                        )
        else:
            for group in destinations.values():
                j = group[0]
                path = paths[j][1]
                for i, (_, g, *_) in enumerate(gy):
                    for a in range(path[5]):
                        owned_store(
                            g,
                            f"node * {ydim} + {path[1] + a * mul} + channel",
                            f"total_y{j}_{i}_{a}",
                        )
        emit("}")
    emit("}")
    common = (shared_size - gradient_size) * itemsize if block_channels else 0
    per_warp = gradient_size * itemsize if block_channels else shared_size * itemsize
    return "\n".join(lines), mul, (common, per_warp)
