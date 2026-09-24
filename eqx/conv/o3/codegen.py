"""Generate sparse CG contractions with shared factors and local reductions."""

from ...kernels.codegen import HEADER


def angular_source(path, mapping, dimensions, shared, roles, cache, lines):
    """Reuse contractions across paths and transposed multilinear terms."""
    start, attr, end, mul1, mul2, dim1, dim2, dim_out, _, _, cg = path

    def load(role, column, channel):
        pointer = mapping[role]
        row = {0: "source[edge]", 3: "edge", 4: "target[edge]"}[role]
        if role == 3 and shared[pointer]:
            row = "0"
        key = pointer, row, column, channel
        if key not in cache:
            name = f"v{len(cache)}"
            value = (
                f"p{pointer}[({row}) * {dimensions[pointer]} + {column} + {channel}]"
            )
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

    needed = set(roles) & {0, 3, 4}
    weight_role = None
    if 1 in roles:
        widths = {0: dim1, 3: dim2, 4: dim_out}
        weight_role = min(needed or widths, key=widths.__getitem__)
        needed.add(weight_role)
    values = {}
    for v in range(mul2):
        for role, dim in ((0, dim1), (3, dim2), (4, dim_out)):
            if role not in needed:
                continue
            # The missing operand must not prevent reuse of an identical
            # contraction with a different destination or cotangent.
            inputs = tuple(
                (i, mapping[i], offset, width)
                for i, offset, width in (
                    (0, start, mul1),
                    (3, attr, mul2),
                    (4, end, mul1),
                )
                if i != role
            )
            for m in range(dim):
                key = ("angular", inputs, 0 if role == 3 else v, role, m, cg)
                if key not in cache:
                    terms = []
                    for a, b, c, coefficient in cg:
                        if {0: a, 3: b, 4: c}[role] != m:
                            continue
                        x = load(0, start + a * mul1, "u") if role != 0 else None
                        y = load(3, attr + b * mul2, str(v)) if role != 3 else None
                        z = load(4, end + c * mul1, "u") if role != 4 else None
                        if role in (0, 3) and {0, 3} <= needed:
                            coefficient_key = "coefficient", coefficient, z
                            if coefficient_key not in cache:
                                name = f"v{len(cache)}"
                                lines.append(
                                    f"const T {name} = T({coefficient:.17g}) * {z};"
                                )
                                cache[coefficient_key] = name
                            terms.append(
                                (cache[coefficient_key], y if role == 0 else x)
                            )
                        else:
                            terms.append(
                                (
                                    f"T({coefficient:.17g})",
                                    product(*[t for t in (x, y, z) if t is not None]),
                                )
                            )
                    name = f"v{len(cache)}"
                    lines.append(f"T {name} = 0;")
                    for coefficient, term in terms:
                        lines.append(f"{name} = fma({coefficient}, {term}, {name});")
                    cache[key] = name
                values[v, role, m] = cache[key]
        if weight_role is not None:
            role = weight_role
            offset, width, channel = {
                0: (start, mul1, "u"),
                3: (attr, mul2, str(v)),
                4: (end, mul1, "u"),
            }[role]
            terms = tuple(
                (values[v, role, m], load(role, offset + m * width, channel))
                for m in range({0: dim1, 3: dim2, 4: dim_out}[role])
            )
            key = "weight", terms
            if key not in cache:
                name = f"v{len(cache)}"
                lines.append(f"T {name} = 0;")
                for a, b in terms:
                    lines.append(f"{name} = fma({a}, {b}, {name});")
                cache[key] = name
            values[v, 1, 0] = cache[key]
    return values


def convolution_source(
    paths, program, dimensions, shared, dtype, owner, outputs, initialize=()
):
    """Emit a path tile, accumulating shared destinations before global writes."""
    mul = paths[0][3]
    args = [f"const T* p{i}" for i in range(len(dimensions))]
    args += [f"T* g{i}" for i in range(outputs)]
    args += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* order",
        "int64_t edges",
        "int64_t tasks",
        "int row_size",
    ]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ", ".join(args) + ") {",
        "const int lane = threadIdx.x % 32;",
        "const int u = int(blockIdx.y) * 32 + lane;",
        f"const bool active = u < {mul};",
        "const int64_t task = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;",
        "if (task >= tasks) return;",
    ]
    node_terms, edge_terms, weight_terms = {}, {}, {}
    cache, body = {}, []
    for mapping, weighted_only, pairs in program:
        needed = {role for role, _ in pairs}
        for path in paths:
            start, attr, end, _, mul2, dim1, dim2, dim_out, weight, factor, _ = path
            if weight < 0 and (weighted_only or needed == {1}):
                continue
            values = angular_source(
                path,
                mapping,
                dimensions,
                shared,
                needed if weight >= 0 else needed - {1},
                cache,
                body,
            )
            for v in range(mul2):
                w = f"T({factor:.17g})"
                if needed & {0, 3, 4} and weight >= 0:
                    pointer = mapping[1]
                    row = "0" if shared[pointer] else "edge"
                    key = "radial", pointer, weight, mul2, v, factor
                    if key not in cache:
                        name = f"v{len(cache)}"
                        body.append(
                            f"const T {name} = active ? T({factor:.17g}) * p{pointer}[{row} * {dimensions[pointer]} + {weight} + u * {mul2} + {v}] : T(0);"
                        )
                        cache[key] = name
                    w = cache[key]
                for role, slot in pairs:
                    pointer = mapping[role]
                    if role in (0, 4):
                        offset, dim = (start, dim1) if role == 0 else (end, dim_out)
                        for m in range(dim):
                            key = slot, role, dimensions[pointer], offset + m * mul
                            node_terms.setdefault(key, []).append(
                                (w, values[v, role, m])
                            )
                    elif role == 3:
                        for m in range(dim2):
                            key = (
                                slot,
                                dimensions[pointer],
                                attr + m * mul2 + v,
                                shared[pointer],
                            )
                            edge_terms.setdefault(key, []).append(
                                (w, values[v, role, m])
                            )
                    elif role == 1 and weight >= 0:
                        key = (
                            slot,
                            dimensions[pointer],
                            weight,
                            mul2,
                            v,
                            shared[pointer],
                        )
                        weight_terms.setdefault(key, []).append(
                            (f"T({factor:.17g})", values[v, 1, 0])
                        )

    if owner >= 0:
        index = "source" if owner == 0 else "target"
        lines += [
            "const int64_t stop = min((task + 1) * row_size, edges);",
            "int64_t end = task * row_size;",
            "while (end < stop) {",
            "const int64_t begin = end;",
            f"const int64_t node = {index}[order[begin]];",
            f"do {{ ++end; }} while (end < stop && {index}[order[end]] == node);",
            f"const bool exclusive = (begin == 0 || {index}[order[begin - 1]] != node) && (end == edges || {index}[order[end]] != node);",
        ]
        lines += [f"T sum{i} = 0;" for i in range(len(node_terms))]
        lines += [
            "for (int64_t position = begin; position < end; ++position) {",
            "const int64_t edge = order[position];",
        ]
    else:
        lines += ["const int64_t edge = task;"]
    lines.extend(body)

    for i, ((slot, role, dim, offset), terms) in enumerate(node_terms.items()):
        name = f"sum{i}"
        if owner < 0:
            lines.append(f"T {name} = 0;")
        for a, b in terms:
            lines.append(f"{name} = fma({a}, {b}, {name});")
        if owner < 0:
            row = "source[edge]" if role == 0 else "target[edge]"
            lines.append(
                f"if (active) atomicAdd(g{slot} + {row} * {dim} + {offset} + u, {name});"
            )
    for (slot, dim, offset, is_shared), terms in edge_terms.items():
        lines += ["{ T value = 0;"]
        for a, b in terms:
            lines.append(f"value = fma({a}, {b}, value);")
        lines.append("value = warp_sum(value);")
        lines.append(
            f"if (lane == 0) atomicAdd(g{slot} + {'0' if is_shared else 'edge'} * {dim} + {offset}, value); }}"
        )
    for (slot, dim, offset, mul2, v, is_shared), terms in weight_terms.items():
        lines += ["{ T value = 0;"]
        for a, b in terms:
            lines.append(f"value = fma({a}, {b}, value);")
        address = f"g{slot}[{'0' if is_shared else 'edge'} * {dim} + {offset} + u * {mul2} + {v}]"
        if is_shared:
            lines.append(f"if (active) atomicAdd(&{address}, value);")
        else:
            lines.append(
                f"if (active) {address} {'=' if slot in initialize else '+='} value;"
            )
        lines.append("}")
    if owner >= 0:
        lines.append("}")
        for i, ((slot, _, dim, offset), _) in enumerate(node_terms.items()):
            address = f"g{slot}[node * {dim} + {offset} + u]"
            lines.append(
                f"if (active) {{ if (exclusive) {address} += sum{i}; else atomicAdd(&{address}, sum{i}); }}"
            )
        lines.append("}")
    lines.append("}")
    return "\n".join(lines)
