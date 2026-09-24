"""Generate channel-parallel sparse CG contractions and their transposes."""

from ...kernels.codegen import HEADER


def angular_source(path, dimensions, shared, roles, cache, lines):
    """Reuse angular inputs and products within a compatible path tile."""
    start, attr, end, mul1, mul2, dim1, dim2, dim_out, _, _, cg = path

    def load(role, column, channel, shared_row=False):
        key = role, column, channel
        if key not in cache:
            name = f"v{len(cache)}"
            row = {0: "source[edge]", 3: "edge", 4: "target[edge]"}[role]
            if shared_row:
                row = "0"
            value = f"p{role}[({row}) * {dimensions[role]} + {column} + {channel}]"
            lines.append(f"const T {name} = active ? {value} : T(0);")
            cache[key] = name
        return cache[key]

    def product(a, b):
        key = tuple(sorted((a, b)))
        if key not in cache:
            name = f"v{len(cache)}"
            lines.append(f"const T {name} = {a} * {b};")
            cache[key] = name
        return cache[key]

    values = {}
    for v in range(mul2):
        # Each transpose removes one angular factor from the scalar product.
        for role, dim in ((0, dim1), (3, dim2), (4, dim_out)):
            if role not in roles and not (role == 4 and (1 in roles or 2 in roles)):
                continue
            for m in range(dim):
                key = ("angular", start, attr, end, v, role, m, cg)
                if key not in cache:
                    terms = []
                    for a, b, c, coefficient in cg:
                        if {0: a, 3: b, 4: c}[role] != m:
                            continue
                        x = load(0, start + a * mul1, "u") if role != 0 else None
                        y = (
                            load(3, attr + b * mul2, str(v), shared[3])
                            if role != 3
                            else None
                        )
                        z = load(4, end + c * mul1, "u") if role != 4 else None
                        terms.append(
                            (
                                coefficient,
                                product(*[t for t in (x, y, z) if t is not None]),
                            )
                        )
                    # Inputs above may have appended new temporaries.
                    name = f"v{len(cache)}"
                    lines.append(f"T {name} = 0;")
                    for coefficient, term in terms:
                        lines.append(
                            f"{name} = fma(T({coefficient:.17g}), {term}, {name});"
                        )
                    cache[key] = name
                values[v, role, m] = cache[key]
        if 1 in roles or 2 in roles:
            terms = [
                (values[v, 4, c], load(4, end + c * mul1, "u")) for c in range(dim_out)
            ]
            name = f"v{len(cache)}"
            lines.append(f"T {name} = 0;")
            for a, b in terms:
                lines.append(f"{name} = fma({a}, {b}, {name});")
            cache["weight", len(cache)] = name
            values[v, 1, 0] = name
    return values


def convolution_source(paths, outputs, dimensions, shared, projected, dtype, owner=-1):
    """Emit one path tile, including fused radial and angular adjoints."""
    roles = set(outputs)
    mul1 = paths[0][3]
    projection_gradient = roles == {2}
    rows = 4 if projection_gradient or owner >= 0 else 1
    args = [f"const T* p{i}" for i in range(5)]
    args += [f"T* g{i}" for i in range(len(outputs))]
    args += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* order",
        "int64_t edges",
    ]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ", ".join(args) + ") {",
        "const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;",
        "const int u = int(blockIdx.y) * 32 + lane;",
        f"const bool channel = u < {mul1};",
    ]
    if projection_gradient:
        # One block contracts 16 edges. A bounded shared tile replaces both
        # an edge-weight-gradient tensor and per-edge parameter atomics.
        lines.append("__shared__ T partial[16 * 32];")
        for path in paths:
            mul2, weight, factor = path[4], path[8], path[9]
            for v in range(mul2):
                lines += [
                    "{",
                    "for (int row = 0; row < 4; ++row) {",
                    "const int local = row * 4 + warp;",
                    "const int64_t edge = int64_t(blockIdx.x) * 16 + local;",
                    "const bool active = channel && edge < edges;",
                ]
                # Inactive edge slots must not load an index or a tensor.
                cache = {}
                values = angular_source(path, dimensions, shared, {2}, cache, lines)
                lines.append(
                    f"partial[local * 32 + lane] = T({factor:.17g}) * {values[v, 1, 0]};"
                )
                lines += ["}", "__syncthreads();"]
                radial_dim = dimensions[1]
                lines += [
                    f"for (int item = threadIdx.x; item < {radial_dim * 32}; item += 128) {{",
                    "const int r = item / 32, c = int(blockIdx.y) * 32 + item % 32;",
                    f"if (c < {mul1}) {{",
                    "T value = 0;",
                    "#pragma unroll",
                    "for (int local = 0; local < 16; ++local) {",
                    "const int64_t edge = int64_t(blockIdx.x) * 16 + local;",
                    "if (edge < edges) {",
                    f"value = fma(p1[{0 if shared[1] else 'edge'} * {radial_dim} + r], partial[local * 32 + item % 32], value);",
                    "}",
                    "}",
                ]
                for i in range(len(outputs)):
                    lines.append(
                        f"atomicAdd(g{i} + r * {dimensions[2]} + {weight} + c * {mul2} + {v}, value);"
                    )
                lines += ["}", "}", "__syncthreads();", "}"]
        lines.append("}")
        return "\n".join(lines), 16

    accumulators = {}
    if owner >= 0:
        role = 0 if owner == 0 else 4
        for i in range(len(outputs)):
            for path in paths:
                start, dim = (path[0], path[5]) if role == 0 else (path[2], path[7])
                for m in range(dim):
                    key = i, start + m * mul1
                    if key not in accumulators:
                        accumulators[key] = f"sum{len(accumulators)}"
        for name in accumulators.values():
            lines.append(f"T {name} = 0;")
        lines.append("int64_t previous = -1;")

    def flush():
        return [
            f"atomicAdd(g{i} + previous * {dimensions[outputs[i]]} + {offset} + u, {name}); {name} = 0;"
            for (i, offset), name in accumulators.items()
        ]

    lines += [
        f"for (int row = 0; row < {rows}; ++row) {{",
        f"const int64_t item = (int64_t(blockIdx.x) * 4 + warp) * {rows} + row;",
        "if (item >= edges) break;",
        "const int64_t edge = order ? order[item] : item;",
        "const bool active = channel;",
    ]
    if owner >= 0:
        lines += [
            f"const int64_t node = {'source' if owner == 0 else 'target'}[edge];",
            "if (previous >= 0 && previous != node && active) {",
            *flush(),
            "}",
            "previous = node;",
        ]
    cache, radial_terms = {}, {}
    for path in paths:
        start, attr, end, _, mul2, dim1, dim2, dim_out, weight, factor, _ = path
        values = angular_source(path, dimensions, shared, roles, cache, lines)
        for v in range(mul2):
            lines.append("{")
            if roles & {0, 3, 4}:
                lines.append("T w = 1;")
                if weight >= 0:
                    lines.append("w = 0;")
                    if projected:
                        lines += [
                            "if (active) {",
                            f"for (int r = 0; r < {dimensions[1]}; ++r) {{",
                            f"w = fma(p1[{0 if shared[1] else 'edge'} * {dimensions[1]} + r], p2[r * {dimensions[2]} + {weight} + u * {mul2} + {v}], w);",
                            "}",
                            "}",
                        ]
                    else:
                        lines.append(
                            f"if (active) w = p1[{0 if shared[1] else 'edge'} * {dimensions[1]} + {weight} + u * {mul2} + {v}];"
                        )
                lines.append(f"w *= T({factor:.17g});")
            for i, role in enumerate(outputs):
                if role in (0, 4):
                    begin, dim = (start, dim1) if role == 0 else (end, dim_out)
                    for m in range(dim):
                        value = f"w * {values[v, role, m]}"
                        if owner >= 0:
                            lines.append(
                                f"{accumulators[i, begin + m * mul1]} += {value};"
                            )
                        else:
                            row = "source[edge]" if role == 0 else "target[edge]"
                            lines.append(
                                f"if (active) atomicAdd(g{i} + {row} * {dimensions[role]} + {begin + m * mul1} + u, {value});"
                            )
                elif role == 3:
                    for m in range(dim2):
                        lines.append(f"{{ T value = warp_sum(w * {values[v, 3, m]});")
                        lines.append(
                            f"if (lane == 0) atomicAdd(g{i} + {0 if shared[3] else 'edge'} * {dimensions[3]} + {attr + m * mul2 + v}, value); }}"
                        )
                elif role == 1 and weight >= 0:
                    if projected:
                        radial_terms.setdefault(i, []).append(
                            (weight, mul2, v, factor, values[v, 1, 0])
                        )
                    else:
                        lines.append(
                            f"if (active) atomicAdd(g{i} + {0 if shared[1] else 'edge'} * {dimensions[1]} + {weight} + u * {mul2} + {v}, T({factor:.17g}) * {values[v, 1, 0]});"
                        )
            lines.append("}")
    # Reduce channel contributions across all paths in this tile before
    # updating a radial feature. The weight adjoints remain in registers.
    for i, terms in radial_terms.items():
        lines += [
            f"for (int r = 0; r < {dimensions[1]}; ++r) {{",
            "T value = 0;",
            "if (active) {",
        ]
        for weight, mul2, v, factor, gradient in terms:
            lines.append(
                f"value = fma(p2[r * {dimensions[2]} + {weight} + u * {mul2} + {v}], T({factor:.17g}) * {gradient}, value);"
            )
        lines += [
            "}",
            "value = warp_sum(value);",
            f"if (lane == 0) atomicAdd(g{i} + {0 if shared[1] else 'edge'} * {dimensions[1]} + r, value);",
            "}",
        ]
    lines.append("}")
    if owner >= 0:
        lines += ["if (previous >= 0 && channel) {", *flush(), "}"]
    lines.append("}")
    return "\n".join(lines), 4 * rows
