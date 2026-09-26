"""CUDA source generation for sparse contractions and frame geometry."""

import math
from collections import Counter, defaultdict
from functools import lru_cache

HEADER = r"""
using int64_t = long long;
using T = SCALAR;
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
"""


def contraction_source(terms, cache, lines):
    """Factor a scalar polynomial and reuse equal or opposite expressions.

    Parameters
    ----------
    terms : iterable of (float, tuple of str)
        Coefficients and scalar operands of each monomial.
    cache : dict
        Expressions already emitted in the current scope.
    lines : list of str
        Generated statements, using ``T`` as the scalar type.
    """
    coefficients = defaultdict(list)
    for coefficient, factors in terms:
        coefficients[tuple(sorted(factors))].append(coefficient)
    terms = tuple(
        (coefficient, factors)
        for factors, values in coefficients.items()
        if (coefficient := math.fsum(values)) != 0
    )
    if not terms:
        return "T(0)"
    # Canonicalize the cache key while preserving the first-use term order.
    canonical = sorted(terms, key=lambda term: term[1])
    sign = -1 if canonical[0][0] < 0 else 1
    key = "contraction", tuple((sign * c, f) for c, f in canonical)
    if key in cache:
        value = cache[key]
        return value if sign == 1 else f"(-{value})"
    if len(terms) == 1 and not terms[0][1]:
        return f"T({terms[0][0]:.17g})"
    if len(terms) == 1 and terms[0][0] == 1 and len(terms[0][1]) == 1:
        return terms[0][1][0]
    counts = Counter(
        factor for _, factors in terms for factor in dict.fromkeys(factors)
    )
    factor, count = counts.most_common(1)[0] if counts else (None, 0)
    if count > 1:
        selected, remaining = [], []
        for coefficient, factors in terms:
            if factor in factors:
                factors = list(factors)
                factors.remove(factor)
                selected.append((coefficient, tuple(factors)))
            else:
                remaining.append((coefficient, factors))
        inner = contraction_source(selected, cache, lines)
        rest = contraction_source(remaining, cache, lines)
        expression = f"fma({factor}, {inner}, {rest})"
    else:
        products = []
        for coefficient, factors in terms:
            value = "T(1)"
            for operand in factors:
                if value == "T(1)":
                    value = operand
                else:
                    product = "contraction_product", *sorted((value, operand))
                    if product not in cache:
                        name = f"v{len(cache)}"
                        lines.append(f"const T {name} = {value} * {operand};")
                        cache[product] = name
                    value = cache[product]
            products.append((coefficient, value))
        name = f"v{len(cache)}"
        cache[key] = name if sign == 1 else f"(-{name})"
        lines.append(f"T {name} = 0;")
        lines.extend(
            f"{name} = fma(T({coefficient:.17g}), {value}, {name});"
            for coefficient, value in products
        )
        return name
    name = f"v{len(cache)}"
    cache[key] = name if sign == 1 else f"(-{name})"
    lines.append(f"const T {name} = {expression};")
    return name


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
    int64_t as0, int64_t as1, int64_t bs0, int64_t bs1,
    int64_t cs0, int64_t cs1) {{
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
    c[edge * cs0 + row * cs1] = value;
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
    if key == "direction_gradient":
        program = ScalarProgram()
        vectors = [program.add("input", 0, i) for i in range(3)]
        torque = [program.add("input", 1, i) for i in range(3)]
        norm = program.constant(0)
        for value in vectors:
            norm = program.add("add", norm, program.add("mul", value, value))
        gradient = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            cross = program.add(
                "add",
                program.add("mul", torque[j], vectors[k]),
                program.add("neg", program.add("mul", torque[k], vectors[j])),
            )
            gradient.append(program.add("div", cross, norm))
        return tuple(program.nodes), (tuple(gradient),), (3, 3)
    if key not in (None, "quaternion", "wigner1"):
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

    if key == "quaternion":
        return tuple(p.nodes), ((w, x, y, z),), (3,)

    def product(a, b):
        return p.add("mul", a, b)

    identity = (
        sum_values([product(q, q) for q in (w, x, y, z)]) if key == "wigner1" else one
    )

    def diag(a, b):
        return p.add(
            "add",
            identity,
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
    if key == "wigner1":
        outputs = ((one, *outputs[0]),)
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
