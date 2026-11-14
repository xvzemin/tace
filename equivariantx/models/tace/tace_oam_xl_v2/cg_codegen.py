"""Generate sparse CG contractions with shared factors and local reductions."""

import os
import re

HOIST_OWNER = os.environ.get("EQX_XL_HOIST_OWNER", "0") == "1"

from equivariantx.ops.common.cuda.codegen import HEADER, contraction_source

FOLD_MIXED_INPUTS = os.environ.get("EQX_XL_FOLD_MIXED_INPUTS", "0") == "1"

POLYNOMIAL_EDGES = os.environ.get("EQX_XL_POLYNOMIAL_EDGES", "0") == "1"

GEOMETRY_COEFFICIENTS = os.environ.get("EQX_XL_GEOMETRY_COEFFICIENTS", "0") == "1"
REUSE_ANGULAR_ADJOINTS = os.environ.get("EQX_XL_REUSE_ANGULAR_ADJOINTS", "0") == "1"


def angular_source(
    path,
    mapping,
    dimensions,
    shared,
    roles,
    cache,
    lines,
    premultiply=None,
    reuse_angular=None,
):
    """Reuse contractions across paths and transposed multilinear terms."""
    start, attr, end, mul1, mul2, dim1, dim2, dim_out, _, _, cg = path
    reuse_angular = REUSE_ANGULAR_ADJOINTS if reuse_angular is None else reuse_angular

    def load(role, column, channel):
        pointer = mapping[role]
        if premultiply is not None and role == premultiply[0]:
            key = ("weighted_input", premultiply, column, channel, path[8])
            if key not in cache:
                pieces = []
                for xp, wp in premultiply[1]:
                    input_row = (
                        "source_node" if role == 0 else "0" if shared[xp] else "edge"
                    )
                    xkey = (xp, input_row, column, channel)
                    if xkey not in cache:
                        name = f"v{len(cache)}"
                        lines.append(
                            f"const T {name}=active?p{xp}[{input_row}*{dimensions[xp]}+{column}+{channel}]:T(0);"
                        )
                        cache[xkey] = name
                    wkey = ("fold_weight", wp, path[8])
                    if wkey not in cache:
                        name = f"v{len(cache)}"
                        edge = "0" if shared[wp] else "edge"
                        lines.append(
                            f"const T {name}=active?p{wp}[{edge}*{dimensions[wp]}+{path[8]}+u]:T(0);"
                        )
                        cache[wkey] = name
                    pieces.append((cache[xkey], cache[wkey]))
                name = f"v{len(cache)}"
                lines.append(f"T {name}=0;")
                for a, b in pieces:
                    lines.append(f"{name}=fma({a},{b},{name});")
                cache[key] = name
            return cache[key]
        row = {0: "source_node", 3: "edge", 4: "target_node"}[role]
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
        candidates = needed or widths
        available = []
        if reuse_angular:
            for candidate in candidates:
                inputs = tuple(
                    (i, mapping[i], offset, width)
                    for i, offset, width in (
                        (0, start, mul1),
                        (3, attr, mul2),
                        (4, end, mul1),
                    )
                    if i != candidate
                )
                if all(
                    (
                        "angular",
                        inputs,
                        0 if candidate == 3 else v,
                        candidate,
                        m,
                        cg,
                        premultiply,
                    )
                    in cache
                    for v in range(mul2)
                    for m in range(widths[candidate])
                ):
                    available.append(candidate)
        weight_role = min(available or candidates, key=widths.__getitem__)
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
                key = (
                    "angular",
                    inputs,
                    0 if role == 3 else v,
                    role,
                    m,
                    cg,
                    premultiply,
                )
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
                        elif GEOMETRY_COEFFICIENTS and role != 3:
                            coefficient_key = "geometry_coefficient", coefficient, y
                            if coefficient_key not in cache:
                                name = f"v{len(cache)}"
                                lines.append(
                                    f"const T {name} = T({coefficient:.17g}) * {y};"
                                )
                                cache[coefficient_key] = name
                            terms.append(
                                (1.0, (cache[coefficient_key], x if role == 4 else z))
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
    indexed=True,
    hoist_owner=None,
    shared_owner=0,
    reuse_angular=None,
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
        args.extend(("int64_t tile", "int channel"))
    declaration = (
        "__device__ __forceinline__" if concurrent else 'extern "C" __global__'
    )
    tile = "tile" if concurrent else "int64_t(blockIdx.x)"
    lines = [
        "" if concurrent else HEADER.replace("SCALAR", dtype),
        f"{declaration} void {function or 'run'}(" + ", ".join(args) + ") {",
        "const int lane = threadIdx.x % 32;",
        f"const int u = {'channel' if concurrent else 'int(blockIdx.y)'} * 32 + lane;",
        (
            "const bool active = true;"
            if mul == 128 and os.environ.get("EQX_XL_LAUNCH_CONTRACT", "0") == "1"
            else f"const bool active = u < {mul};"
        ),
        f"const int64_t task = {tile} * (blockDim.x / 32) + threadIdx.x / 32;",
        "if (task >= tasks) return;",
    ]
    node_terms, edge_terms, weight_terms = {}, {}, {}
    cache, body = {}, []
    scheduled = []
    groups = {}
    if (
        FOLD_MIXED_INPUTS
        and normalization is None
        and all(p[4] == 1 and p[8] >= 0 for p in paths)
    ):
        for mapping, weighted_only, pairs in program:
            for role, slot in pairs:
                if role in (0, 4):
                    common = mapping[3] if role == 4 else mapping[4]
                    groups.setdefault((role, slot, common, weighted_only), []).append(
                        mapping
                    )
        merged = {key: values for key, values in groups.items() if len(values) > 1}
        for mapping, weighted_only, pairs in program:
            keep = tuple(
                (role, slot)
                for role, slot in pairs
                if role not in (0, 4)
                or (role, slot, mapping[3] if role == 4 else mapping[4], weighted_only)
                not in merged
            )
            if keep:
                scheduled.append((mapping, weighted_only, keep, None))
        for (role, slot, _, weighted_only), mappings in merged.items():
            folded_role = 0 if role == 4 else 3
            folded = (folded_role, tuple((m[folded_role], m[1]) for m in mappings))
            scheduled.append((mappings[0], weighted_only, ((role, slot),), folded))
    else:
        scheduled = [(*term, None) for term in program]
    for mapping, weighted_only, pairs, folded in scheduled:
        needed = {role for role, _ in pairs}
        for path in paths:
            start, attr, end, _, mul2, dim1, dim2, dim_out, weight, factor, _ = path
            if weight < 0 and (weighted_only or needed == {1}):
                continue
            angular = angular_source
            if normalization is not None:
                from equivariantx.ops.common.cuda.o3.harmonics import (
                    angular_source as angular,
                )
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
                **({"premultiply": folded} if folded is not None else {}),
                **({"reuse_angular": reuse_angular} if normalization is None else {}),
            )
            for v in range(mul2):
                w = f"T({factor:.17g})"
                if needed - {1, 2} and weight >= 0 and folded is None:
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
                            if (v, role, m) not in values:
                                continue
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

    if (
        POLYNOMIAL_EDGES
        and normalization is None
        and all(p[4] == 1 and p[8] >= 0 for p in paths)
        and all(role in (0, 1, 3, 4) for _, _, pairs in program for role, _ in pairs)
    ):
        from .polynomial_edge import edge_polynomials

        cache, body, node_terms, edge_terms, weight_terms = edge_polynomials(
            paths, program, dimensions, shared
        )

    if "transverse_adjoints" in cache:
        from equivariantx.ops.common.cuda.o2_o3.transverse import (
            coupling_adjoint_source,
        )

        pending, groups = cache["transverse_adjoints"], {}
        for key, terms in edge_terms.items():
            remaining = []
            slot, width, column, shared_output = key
            for weight, value in terms:
                if value not in pending:
                    remaining.append((weight, value))
                    continue
                request, axis = pending[value]
                if axis == 0:
                    group = slot, width, shared_output, request[3]
                    groups.setdefault(group, []).append((*request, weight))
            edge_terms[key] = remaining
        for (slot, width, shared_output, _), requests in groups.items():
            gradient = coupling_adjoint_source(requests, cache, body)
            for column, value in enumerate(gradient):
                key = slot, width, column, shared_output
                edge_terms[key].append(("T(1)", value))
        groups = {}
        for key, terms in node_terms.items():
            remaining = []
            slot, role, width, column = key
            for weight, value in terms:
                if value not in pending:
                    remaining.append((weight, value))
                    continue
                request, axis = pending[value]
                if axis == 0:
                    group = slot, role, width, column, request[3:5]
                    groups.setdefault(group, []).append((*request, weight))
            node_terms[key] = remaining
        for (slot, role, width, offset, _), requests in groups.items():
            gradient = coupling_adjoint_source(
                requests, cache, body, feature_adjoint=True
            )
            for column, value in enumerate(gradient):
                key = slot, role, width, offset + column * mul
                node_terms[key].append(("T(1)", value))

    hoisted = []
    if owner >= 0 and indexed and (HOIST_OWNER if hoist_owner is None else hoist_owner):
        invariant = "source_node" if owner == 0 else "target_node"
        known = set()
        remainder = []
        shared_count = 0
        for line in body:
            match = re.fullmatch(r"const T (v\d+) = (.*);", line)
            if match:
                expression = match[2]
                dependencies = set(re.findall(r"\bv\d+\b", expression))
                other = "target_node" if owner == 0 else "source_node"
                if shared_owner:
                    if (
                        invariant in expression
                        and re.search(r"\bp\d+\[", expression)
                        and not re.search(r"\bedge\b", expression)
                        and other not in expression
                    ):
                        address = f"owner_values[{shared_count}*blockDim.x+threadIdx.x]"
                        hoisted.append(
                            f"{address} = {expression.replace(invariant, 'node')};"
                        )
                        remainder.append(f"const T {match[1]} = {address};")
                        shared_count += 1
                        continue
                    remainder.append(line)
                    continue
                if (
                    (invariant in expression or dependencies)
                    and dependencies <= known
                    and not re.search(r"\bedge\b", expression)
                    and other not in expression
                ):
                    hoisted.append(line.replace(invariant, "node"))
                    known.add(match[1])
                    continue
            remainder.append(line)
        body = remainder
        if shared_owner and shared_count:
            lines.append(f"// EQX_OWNER_VALUES={shared_count}")
            lines.append(
                "extern __shared__ "
                + ("volatile " if shared_owner == 2 else "")
                + "T owner_values[];"
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
        lines.extend(hoisted)
        lines += [
            "for (int64_t position = begin; position < end; ++position) {",
            "const int64_t edge = order[position];",
        ]
    else:
        lines += ["const int64_t edge = task;"]
    lines += [
        f"const int64_t source_node = {'source[edge]' if indexed else 'edge'};",
        f"const int64_t target_node = {'target[edge]' if indexed else 'edge'};",
    ]
    body_start = len(lines)
    lines.extend(body)
    sinks = []
    for i, ((slot, role, dim, offset), terms) in enumerate(node_terms.items()):
        sinks.append((len(lines), role))
        name = f"sum{i}"
        reduced = owner >= 0 and role == (0 if owner == 0 else 4)
        if not reduced:
            lines.append(f"T {name} = 0;")
        for a, b in terms:
            lines.append(f"{name} = fma({a}, {b}, {name});")
        if not reduced:
            row = "source_node" if role == 0 else "target_node"
            lines.append(
                f"if (active) atomicAdd(g{slot} + {row} * {dim} + {offset} + u, {name});"
            )
    for (slot, dim, offset, is_shared), terms in edge_terms.items():
        sinks.append((len(lines), 3))
        lines += ["{ T value = 0;"]
        for a, b in terms:
            lines.append(f"value = fma({a}, {b}, value);")
        lines.append("value = warp_sum(value);")
        lines.append(
            f"if (lane == 0) atomicAdd(g{slot} + {'0' if is_shared else 'edge'} * {dim} + {offset}, value); }}"
        )
    for (slot, dim, offset, mul2, v, is_shared), terms in weight_terms.items():
        sinks.append((len(lines), 1))
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
    mode = os.environ.get("EQX_XL_STREAM_ANGULAR_OUTPUTS", "off")
    if mode == "auto":
        from .angular_schedule import schedule_mode

        mode = schedule_mode(mode, program)
    if mode != "off" and sinks and normalization is None:
        from .angular_schedule import stream_body

        ends = [start for start, _ in sinks[1:]] + [len(lines)]
        chunks = [(lines[start:end], role) for (start, role), end in zip(sinks, ends)]
        lines = lines[:body_start] + stream_body(body, chunks, mode)
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
    indexed=True,
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
                indexed,
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
    # Equal channel widths retain a rectangular grid. Unequal widths use
    # only their own channel tiles, without padding to the largest phase.
    uniform = len({(paths[0][3] + 31) // 32 for paths, _, _ in phases}) == 1
    layouts = [
        (owner >= 0, 1 if uniform else (paths[0][3] + 31) // 32)
        for paths, _, owner in phases
    ]
    for grouped, channels in dict.fromkeys(layouts):
        indices = [
            i for i, layout in enumerate(layouts) if layout == (grouped, channels)
        ]
        rows = "row_size" if grouped else "1"
        tiles = len(indices) * channels
        channel = (
            "int(blockIdx.y)"
            if uniform
            else f"((int64_t(blockIdx.x) - first) / {len(indices)}) % {channels}"
        )
        body += [
            "{",
            f"const int64_t tasks = (edges + {rows} - 1) / {rows};",
            "const int64_t blocks = (tasks + blockDim.x / 32 - 1) / (blockDim.x / 32);",
            f"if (int64_t(blockIdx.x) < first + blocks * {tiles}) {{",
            f"const int64_t tile = (int64_t(blockIdx.x) - first) / {tiles};",
            f"const int channel = {channel};",
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
                f"case {local}: phase{i}({', '.join(names)}, source, target, {order}, edges, tasks, {rows}, tile, channel); break;"
            )
        body += ["}", "return; }", f"first += blocks * {tiles}; }}"]
    body.append("}")
    return "\n".join(body)
