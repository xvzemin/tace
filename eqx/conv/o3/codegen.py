"""Generate sparse CG contractions with shared factors and local reductions."""

from ...kernels.codegen import HEADER
from ..angular import contraction_source


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
                                    coefficient,
                                    tuple(t for t in (x, y, z) if t is not None),
                                )
                            )
                    if role in (0, 3) and {0, 3} <= needed:
                        name = f"v{len(cache)}"
                        lines.append(f"T {name} = 0;")
                        for coefficient, term in terms:
                            lines.append(
                                f"{name} = fma({coefficient}, {term}, {name});"
                            )
                    else:
                        name = contraction_source(terms, cache, lines)
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
    paths,
    program,
    dimensions,
    shared,
    dtype,
    owner,
    outputs,
    initialize=(),
    function=None,
    atomic_weights=(),
    normalization=None,
    use_generators=False,
    angular_derivatives=False,
    direction_offset=0,
):
    """Emit a path tile, accumulating shared destinations before global writes."""
    mul = paths[0][3]
    concurrent = function is not None
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
    if concurrent:
        args.append("int64_t tile")
    declaration = (
        "__device__ __forceinline__" if concurrent else 'extern "C" __global__'
    )
    tile = "tile" if concurrent else "int64_t(blockIdx.x)"
    lines = [
        "" if concurrent else HEADER.replace("SCALAR", dtype),
        f"{declaration} void {function or 'run'}(" + ", ".join(args) + ") {",
        "const int lane = threadIdx.x % 32;",
        "const int u = int(blockIdx.y) * 32 + lane;",
        f"const bool active = u < {mul};",
        f"const int64_t task = {tile} * (blockDim.x / 32) + threadIdx.x / 32;",
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
            angular = angular_source
            if normalization is not None:
                from .harmonics import angular_source as angular
            values = angular(
                path,
                mapping,
                dimensions,
                shared,
                needed if weight >= 0 else needed - {1},
                cache,
                body,
                *(
                    ()
                    if normalization is None
                    else (
                        normalization,
                        use_generators,
                        angular_derivatives,
                        direction_offset,
                    )
                ),
            )
            for v in range(mul2):
                w = f"T({factor:.17g})"
                if needed - {1, 2} and weight >= 0:
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
                        for m in range(dim2 if normalization is None else 1):
                            key = (
                                slot,
                                dimensions[pointer],
                                attr + m * mul2 + v,
                                shared[pointer],
                            )
                            edge_terms.setdefault(key, []).append(
                                (w, values[v, role, m])
                            )
                    elif role >= 6:
                        for m in range(3):
                            key = slot, dimensions[pointer], m, shared[pointer]
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
        lines += [
            f"T sum{i} = 0;"
            for i, (_, role, _, _) in enumerate(node_terms)
            if role == (0 if owner == 0 else 4)
        ]
        lines += [
            "for (int64_t position = begin; position < end; ++position) {",
            "const int64_t edge = order[position];",
        ]
    else:
        lines += ["const int64_t edge = task;"]
    lines.extend(body)

    for i, ((slot, role, dim, offset), terms) in enumerate(node_terms.items()):
        name = f"sum{i}"
        reduced = owner >= 0 and role == (0 if owner == 0 else 4)
        if not reduced:
            lines.append(f"T {name} = 0;")
        for a, b in terms:
            lines.append(f"{name} = fma({a}, {b}, {name});")
        if not reduced:
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
        if is_shared or slot in atomic_weights:
            lines.append(f"if (active) atomicAdd(&{address}, value);")
        else:
            lines.append(
                f"if (active) {address} {'=' if slot in initialize else '+='} value;"
            )
        lines.append("}")
    if owner >= 0:
        lines.append("}")
        for i, ((slot, role, dim, offset), _) in enumerate(node_terms.items()):
            if role != (0 if owner == 0 else 4):
                continue
            address = f"g{slot}[node * {dim} + {offset} + u]"
            if concurrent or any(
                other_slot == slot and other_role != role
                for other_slot, other_role, _, _ in node_terms
            ):
                lines.append(f"if (active) atomicAdd(&{address}, sum{i});")
            else:
                lines.append(
                    f"if (active) {{ if (exclusive) {address} += sum{i}; else atomicAdd(&{address}, sum{i}); }}"
                )
        lines.append("}")
    lines.append("}")
    return "\n".join(lines)


def fused_source(
    phases,
    dimensions,
    shared,
    dtype,
    outputs,
    initialize,
    normalization=None,
    use_generators=False,
    angular_derivatives=False,
    direction_offset=0,
):
    """Execute independent path tiles in one grid with shared launch operands."""
    header = HEADER.replace("SCALAR", dtype)
    args = [f"const T* p{i}" for i in range(len(dimensions))]
    args += [f"T* g{i}" for i in range(outputs)]
    names = [f"p{i}" for i in range(len(dimensions))] + [
        f"g{i}" for i in range(outputs)
    ]
    writers = {}
    for i, (paths, program, _) in enumerate(phases):
        for _, _, pairs in program:
            for role, slot in pairs:
                if role == 1:
                    for path in paths:
                        if path[8] >= 0:
                            writers.setdefault((slot, path[8]), set()).add(i)
    atomic_weights = {slot for (slot, _), owners in writers.items() if len(owners) > 1}
    body = [header]
    for i, (paths, program, owner) in enumerate(phases):
        body.append(
            convolution_source(
                paths,
                program,
                dimensions,
                shared,
                dtype,
                owner,
                outputs,
                initialize,
                f"phase{i}",
                atomic_weights,
                normalization,
                use_generators,
                angular_derivatives,
                direction_offset,
            )
        )
    args += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* source_order",
        "const int64_t* target_order",
        "int64_t edges",
        "int row_size",
    ]
    body += [
        'extern "C" __global__ void run(' + ", ".join(args) + ") {",
        "int64_t first = 0;",
    ]
    for grouped in (True, False):
        indices = [
            i for i, (_, _, owner) in enumerate(phases) if (owner >= 0) == grouped
        ]
        if not indices:
            continue
        rows = "row_size" if grouped else "1"
        body += [
            "{",
            f"const int64_t tasks = (edges + {rows} - 1) / {rows};",
            "const int64_t blocks = (tasks + blockDim.x / 32 - 1) / (blockDim.x / 32);",
            f"if (int64_t(blockIdx.x) < first + blocks * {len(indices)}) {{",
            f"const int64_t tile = (int64_t(blockIdx.x) - first) / {len(indices)};",
            f"switch ((int64_t(blockIdx.x) - first) % {len(indices)}) {{",
        ]
        for local, i in enumerate(indices):
            owner = phases[i][2]
            order = (
                "source_order"
                if owner == 0
                else "target_order"
                if owner == 1
                else "nullptr"
            )
            body.append(
                f"case {local}: phase{i}({', '.join(names)}, source, target, {order}, edges, tasks, {rows}, tile); break;"
            )
        body += ["}", "return; }", f"first += blocks * {len(indices)}; }}"]
    body.append("}")
    return "\n".join(body)
