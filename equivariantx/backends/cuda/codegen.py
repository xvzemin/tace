"""CUDA source generation for sparse contractions and frame geometry."""

import math
from collections import Counter, defaultdict
from functools import lru_cache

from ...ops.scalar import alignment_program

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
