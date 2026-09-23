"""Generate scalar CUDA contractions from shared multilinear schedules."""

import math
from functools import lru_cache

from .schedule import rotation_groups

HEADER = r"""
using int64_t = long long;
using T = SCALAR;
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
"""


@lru_cache(maxsize=512)
def convolution_source(
    paths, operations, dimensions, shared, inputs, outputs, dtype, owner
):
    """Emit shared input rotations and path/channel tiles of output rotations."""
    xp, yp, messages, adjoints, qs, accumulators, gx, gdi, gy, gdo, gr, gs = operations
    xdim, ydim, ddim, rdim, sdim = dimensions
    mul = paths[0][1][2]
    input_groups, output_groups = rotation_groups([(path, cg) for _, path, cg in paths])
    input_paths = [paths[group[0]][1] for group in input_groups]
    input_index = {j: i for i, group in enumerate(input_groups) for j in group}
    support = [{b: (a, c) for a, b, c in cg} for _, _, cg in paths]
    # Wigner entries are identical across the channel lanes. Stage matrices
    # once per warp instead of retaining a full matrix in every lane's registers.
    matrices = {}
    for path in input_paths:
        for pd in (*[pd for _, pd in xp], *[pd for pd, *_ in gx]):
            matrices[pd, "ei", path[6], path[4]] = None
    # C has one nonzero per supported order. Fold it into the output Wigner
    # rows once per warp, then reuse those rows across channels and transposes.
    coupled_offsets, coupled_entries = {}, {}
    coupled_size = 0
    for j, (_, path, _) in enumerate(paths):
        for pd in (*[pd for _, pd in yp], *[pd for pd, *_ in gy]):
            for b, (_, c) in support[j].items():
                key = pd, path[7] + b * path[5], path[5], c
                if key not in coupled_entries:
                    coupled_entries[key] = coupled_size
                    coupled_size += path[5]
                coupled_offsets[pd, j, b] = coupled_entries[key]
    gradients = {}
    gradient_size = 0
    for _, path, _ in paths:
        for _, g, *_ in gdo:
            key = g, path[7], path[5]
            if key not in gradients:
                gradients[key] = gradient_size
                gradient_size += path[5] ** 2
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
        "const int64_t* tasks",
        "int64_t edges",
        "int64_t task_count",
    ]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ",".join(args) + ") {",
        "const int lane = threadIdx.x & 31;",
        "const int64_t item = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;",
        "const int channel = blockIdx.y * 32 + lane;",
        f"const bool active = channel < {mul};",
    ]
    emit = lines.append
    if staging and matrix_size:
        emit(f"__shared__ T storage[4 * {matrix_size}];")
        emit(f"volatile T* matrix = storage + (threadIdx.x / 32) * {matrix_size};")
    if coupled_staging and coupled_size:
        emit(f"__shared__ T coupled_storage[4 * {coupled_size}];")
        emit(
            f"volatile T* coupled_matrix = coupled_storage + (threadIdx.x / 32) * {coupled_size};"
        )
    if gradient_size:
        emit(f"__shared__ T gradient_storage[4 * {gradient_size}];")
        emit(
            f"T* matrix_gradient = gradient_storage + (threadIdx.x / 32) * {gradient_size};"
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
                    if couplings is None:
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

    def store(g, offset, value, reduced=False, private=False):
        if reduced:
            emit(
                f"{{ T reduced_value = warp_sum({value}); if (lane == 0) atomicAdd(g{g} + {offset}, reduced_value); }}"
            )
        elif private:
            emit(f"if (active) g{g}[{offset}] += {value};")
        else:
            emit(f"if (active) atomicAdd(g{g} + {offset}, {value});")

    if owner >= 0:
        emit("if (item >= task_count) return;")
        emit("const int64_t node = tasks[4 * item];")
        emit("const int64_t begin = tasks[4 * item + 1], end = tasks[4 * item + 2];")
        emit("const bool exclusive = tasks[4 * item + 3];")
        if owner == 0:
            for k, path in enumerate(input_paths):
                for i, _ in enumerate(gx):
                    for a in range(path[4]):
                        emit(f"T total_x{k}_{i}_{a} = 0;")
        else:
            for j, (_, path, _) in enumerate(paths):
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
    if staging and matrix_size:
        for (pd, e, offset, size), location in matrices.items():
            emit(
                f"for (int k = lane; k < {size * size}; k += 32) "
                f"matrix[{location} + k] = p{pd}[{e} * {ddim} + {offset} + k];"
            )
    if coupled_staging and coupled_size:
        for (pd, offset, size, c), location in coupled_entries.items():
            emit(
                f"for (int k = lane; k < {size}; k += 32) "
                f"coupled_matrix[{location} + k] = T({c:.17g}) * p{pd}[eo * {ddim} + {offset} + k];"
            )
    if (staging and matrix_size) or (coupled_staging and coupled_size) or gradient_size:
        emit("__syncwarp();")
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
            destinations = {f"out{j}_{i}_": j for j in group}

            def consume(name, a):
                j = destinations[name]
                if owner == 1:
                    emit(f"total_y{j}_{i}_{a} += {name}{a};")
                else:
                    store(
                        g,
                        f"dst * {ydim} + {paths[j][1][1] + a * mul} + channel",
                        f"{name}{a}",
                    )

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
                    for j in group
                ],
                pd,
                "eo",
                ostart,
                odim,
                True,
                consume,
                destinations,
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
                        location = gradients[g, ostart, odim] + a * odim + b
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
                    )
            for g, *terms in gs:
                value = " + ".join(
                    f"q{terms[t]} * {weight_value(j, terms[t + 1], terms[t + 2])}"
                    for t in range(0, len(terms), 3)
                )
                store(g, f"es * {sdim} + {harmonic}", value, True)
            emit("}")
        emit("}")
    if gradient_size:
        emit("__syncwarp();")
        for (g, offset, size), location in gradients.items():
            emit(
                f"for (int k = lane; k < {size * size}; k += 32) "
                f"atomicAdd(g{g} + eo * {ddim} + {offset} + k, matrix_gradient[{location} + k]);"
            )
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
                    store(
                        g,
                        f"ei * {ddim} + {dstart + a * dim + b}",
                        f"dx{k}_{accumulator}_{a} * ox{k}_{i}_{b}",
                        True,
                    )
    if owner >= 0:
        if (staging and matrix_size) or coupled_size or gradient_size:
            emit("__syncwarp();")
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
            for j, (_, path, _) in enumerate(paths):
                for i, (_, g, *_) in enumerate(gy):
                    for a in range(path[5]):
                        owned_store(
                            g,
                            f"node * {ydim} + {path[1] + a * mul} + channel",
                            f"total_y{j}_{i}_{a}",
                        )
    emit("}")
    return "\n".join(lines), mul


@lru_cache(maxsize=128)
def rotation_source(widths, terms, dtype, output):
    """Generate a sparse bilinear degree contraction or its transpose."""
    a, b = [width for i, width in enumerate(widths) if i != output]
    c = widths[output]
    return (
        HEADER.replace("SCALAR", dtype)
        + f"""
extern "C" __global__ void run(const T* a, const T* b, T* c,
    const int* indices, const T* coefficients, int64_t edges,
    int64_t as0, int64_t as1, int64_t bs0, int64_t bs1) {{
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= edges * {c}) return;
    const int64_t edge = index / {c}, row = index % {c};
    T value = 0;
    for (int k = 0; k < {terms}; ++k) {{
        const int64_t term = row * {terms} + k;
        const T cg = coefficients[term];
        if (cg != T(0)) value = fma(cg * a[edge * as0 + indices[2 * term] * as1],
                                  b[edge * bs0 + indices[2 * term + 1] * bs1], value);
    }}
    c[index] = value;
}}
"""
    )


class ScalarProgram:
    """A small expression graph for the alignment formula and its transposes."""

    def __init__(self, nodes=()):
        self.nodes = list(nodes)
        self.indices = {node: i for i, node in enumerate(nodes)}

    def constant(self, value):
        return self.add("constant", float(value))

    def add(self, op, *args):
        if op in ("add", "mul"):
            args = tuple(sorted(args))
        constants = (
            [self.nodes[i][1] if self.nodes[i][0] == "constant" else None for i in args]
            if op not in ("constant", "input")
            else []
        )
        if op == "add":
            if constants[0] == 0:
                return args[1]
            if constants[1] == 0:
                return args[0]
        if op == "mul":
            if 0 in constants:
                return self.constant(0)
            if constants[0] == 1:
                return args[1]
            if constants[1] == 1:
                return args[0]
        if op == "div" and constants[1] == 1:
            return args[0]
        if constants and all(value is not None for value in constants):
            a = constants[0]
            b = constants[1] if len(constants) > 1 else None
            value = {
                "add": lambda: a + b,
                "mul": lambda: a * b,
                "div": lambda: a / b,
                "neg": lambda: -a,
                "sqrt": lambda: math.sqrt(a),
                "exp": lambda: math.exp(a),
                "lt": lambda: a < b,
                "le": lambda: a <= b,
                "select": lambda: constants[1] if a else constants[2],
            }[op]()
            return self.constant(value)
        node = (op, *args)
        if node not in self.indices:
            self.indices[node] = len(self.nodes)
            self.nodes.append(node)
        return self.indices[node]

    def transpose(self, outputs, widths, active):
        """Append a reverse pass; the resulting graph remains differentiable."""
        end = len(self.nodes)
        gradients = {}
        zero, half = self.constant(0), self.constant(0.5)

        def accumulate(index, value):
            gradients[index] = self.add("add", gradients.get(index, zero), value)

        for i, group in enumerate(outputs):
            for j, node in enumerate(group):
                accumulate(node, self.add("input", len(widths) + i, j))
        for i in range(end - 1, -1, -1):
            if i not in gradients:
                continue
            g = gradients[i]
            op, *args = self.nodes[i]
            if op == "add":
                for a in args:
                    accumulate(a, g)
            elif op == "mul":
                a, b = args
                accumulate(a, self.add("mul", g, b))
                accumulate(b, self.add("mul", g, a))
            elif op == "div":
                a, b = args
                accumulate(a, self.add("div", g, b))
                accumulate(
                    b, self.add("neg", self.add("div", self.add("mul", g, i), b))
                )
            elif op == "neg":
                accumulate(args[0], self.add("neg", g))
            elif op == "sqrt":
                accumulate(args[0], self.add("div", self.add("mul", half, g), i))
            elif op == "exp":
                accumulate(args[0], self.add("mul", g, i))
            elif op == "select":
                condition, a, b = args
                accumulate(a, self.add("select", condition, g, zero))
                accumulate(b, self.add("select", condition, zero, g))
        result = tuple(
            tuple(
                gradients.get(self.indices["input", i, j], zero)
                for j in range(widths[i])
            )
            for i in active
        )
        return result, widths + tuple(map(len, outputs))


@lru_cache(maxsize=128)
def alignment_program(key, dtype):
    """Represent the existing quaternion alignment, without changing its branches."""
    if key is not None:
        parent, active = key
        nodes, outputs, widths = alignment_program(parent, dtype)
        program = ScalarProgram(nodes)
        outputs, widths = program.transpose(outputs, widths, active)
        return tuple(program.nodes), outputs, widths
    p = ScalarProgram()
    zero, one, two, half = map(p.constant, (0, 1, 2, 0.5))
    eps2 = p.constant(1e-14)
    machine_eps = p.constant(2 ** (-23 if dtype == "float" else -52))

    def sum_values(values):
        result = zero
        for value in values:
            result = p.add("add", result, value)
        return result

    def normalize(values):
        norm = p.add(
            "sqrt", p.add("add", sum_values([p.add("mul", v, v) for v in values]), eps2)
        )
        return [p.add("div", v, norm) for v in values]

    x, y, z = normalize([p.add("input", 0, i) for i in range(3)])
    positive = normalize([p.add("add", one, y), p.add("neg", z), zero, x])
    negative = normalize([p.add("neg", z), p.add("add", one, p.add("neg", y)), x, zero])
    t = p.add("mul", half, p.add("add", y, one))
    t = p.add(
        "select",
        p.add("lt", t, zero),
        zero,
        p.add("select", p.add("lt", one, t), one, t),
    )

    def smooth_exp(value):
        clamped = p.add("select", p.add("lt", value, machine_eps), machine_eps, value)
        return p.add("exp", p.add("neg", p.add("div", one, clamped)))

    left, right = smooth_exp(t), smooth_exp(p.add("add", one, p.add("neg", t)))
    blend = p.add("div", left, p.add("add", left, right))
    blend = p.add(
        "select",
        p.add("le", t, zero),
        zero,
        p.add("select", p.add("le", one, t), one, blend),
    )
    dot = sum_values([p.add("mul", a, b) for a, b in zip(positive, negative)])
    positive = [
        p.add("select", p.add("lt", dot, zero), p.add("neg", q), q) for q in positive
    ]
    w, x, y, z = normalize(
        [
            p.add(
                "add",
                p.add("mul", p.add("add", one, p.add("neg", blend)), a),
                p.add("mul", blend, b),
            )
            for a, b in zip(negative, positive)
        ]
    )

    def product(a, b):
        return p.add("mul", a, b)

    def diag(a, b):
        return p.add(
            "add",
            one,
            p.add("neg", product(two, p.add("add", product(a, a), product(b, b)))),
        )

    def off(a, b, c, d, sign):
        second = product(c, d)
        if sign < 0:
            second = p.add("neg", second)
        return product(two, p.add("add", product(a, b), second))

    outputs = (
        (
            diag(y, z),
            off(x, y, w, z, -1),
            off(x, z, w, y, 1),
            off(x, y, w, z, 1),
            diag(x, z),
            off(y, z, w, x, -1),
            off(x, z, w, y, -1),
            off(y, z, w, x, 1),
            diag(x, y),
        ),
    )
    return tuple(p.nodes), outputs, (3,)


@lru_cache(maxsize=128)
def alignment_source(key, dtype):
    """Emit only expressions needed by the requested alignment derivatives."""
    nodes, outputs, widths = alignment_program(key, dtype)
    needed = set(node for group in outputs for node in group)
    for i in range(len(nodes) - 1, -1, -1):
        if i in needed and nodes[i][0] not in ("input", "constant"):
            needed.update(nodes[i][1:])
    args = [f"const T* p{i}" for i in range(len(widths))]
    args += [f"T* g{i}" for i in range(len(outputs))] + ["int64_t edges"]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ",".join(args) + ") {",
        "const int64_t edge = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;",
        "if (edge >= edges) return;",
    ]
    for i, (op, *args) in enumerate(nodes):
        if i not in needed:
            continue
        if op == "constant":
            value = f"T({args[0]:.17g})"
        elif op == "input":
            value = f"p{args[0]}[edge * {widths[args[0]]} + {args[1]}]"
        else:
            a = f"v{args[0]}"
            b = f"v{args[1]}" if len(args) > 1 else ""
            if op == "select":
                value = f"{a} ? {b} : v{args[2]}"
            elif op in ("sqrt", "exp"):
                value = f"{op}({a})"
            elif op == "neg":
                value = f"-{a}"
            else:
                value = (
                    f"{a} "
                    + {"add": "+", "mul": "*", "div": "/", "lt": "<", "le": "<="}[op]
                    + f" {b}"
                )
        lines.append(f"const T v{i} = {value};")
    for i, group in enumerate(outputs):
        for j, node in enumerate(group):
            lines.append(f"g{i}[edge * {len(group)} + {j}] = v{node};")
    lines.append("}")
    return "\n".join(lines)
