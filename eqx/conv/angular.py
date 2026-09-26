"""Fixed generator contractions in the real spherical-harmonic basis."""

import math
from collections import Counter
from functools import lru_cache

import torch
from e3nn import o3


@lru_cache(maxsize=64)
def generators(degree):
    """Return the three real skew rotation generators in float64 on the CPU."""
    if not degree:
        return torch.zeros(3, 1, 1, dtype=torch.float64, device="cpu")
    result = -math.sqrt(degree * (degree + 1) * (2 * degree + 1)) * o3.wigner_3j(
        degree, 1, degree, dtype=torch.float64, device="cpu"
    ).permute(1, 2, 0)
    return result * result[1, degree - 1, degree + 1].sign()


@lru_cache(maxsize=128)
def generator_scale(degree, harmonic, normalization):
    """Return the CG normalization of a degree-one/two generator polynomial."""
    casimir = degree * (degree + 1)
    matrix = generators(degree) / math.sqrt(casimir)
    cg = o3.wigner_3j(degree, harmonic, degree, dtype=torch.float64, device="cpu")
    if harmonic == 1:
        scale = float(cg[degree - 1, 1, degree + 1] / matrix[1, degree + 1, degree - 1])
    elif harmonic == 2:
        scale = 3 * float(cg[degree, 2, degree])
    else:
        raise ValueError("Generator polynomials require harmonic degree one or two.")
    if normalization != "norm":
        scale *= math.sqrt(2 * harmonic + 1)
    if normalization == "integral":
        scale /= math.sqrt(4 * math.pi)
    return scale


def generator_action(degree, vector, features, cache, lines, normalized=False):
    """Apply a sparse generator to a vector of scalar expressions."""
    matrix = generators(degree)
    if normalized:
        matrix = matrix / math.sqrt(degree * (degree + 1))
    result = []
    for i in range(2 * degree + 1):
        key = "generator", degree, normalized, i, vector, features
        if key not in cache:
            terms = []
            for a, j in matrix[:, i, :].nonzero().tolist():
                if vector[a] == "T(0)":
                    continue
                coefficient = float(matrix[a, i, j])
                coefficient_key = "generator_coefficient", coefficient, vector[a]
                if coefficient_key not in cache:
                    name = f"v{len(cache)}"
                    lines.append(
                        f"const T {name} = T({coefficient:.17g}) * {vector[a]};"
                    )
                    cache[coefficient_key] = name
                terms.append((cache[coefficient_key], features[j]))
            name = f"v{len(cache)}"
            cache[key] = name
            lines.append(f"T {name} = 0;")
            lines.extend(f"{name} = fma({a}, {b}, {name});" for a, b in terms)
        result.append(cache[key])
    return tuple(result)


def generator_source(degree, harmonic, vector, features, cache, lines, normalization):
    """Emit a degree-one/two coupling without dense generator powers."""
    scale = generator_scale(degree, harmonic, normalization)

    def variable(key, expression):
        if key not in cache:
            name = f"v{len(cache)}"
            lines.append(f"const T {name} = {expression};")
            cache[key] = name
        return cache[key]

    values = features
    for _ in range(harmonic):
        values = generator_action(degree, vector, values, cache, lines, normalized=True)
    if harmonic == 2:
        norm = variable(
            ("norm", vector),
            f"({vector[0]} * {vector[0]} + {vector[1]} * {vector[1]} + {vector[2]} * {vector[2]}) / T(3)",
        )
        values = tuple(
            variable(("quadrupole", a, b, norm), f"fma({norm}, {b}, {a})")
            for a, b in zip(values, features)
        )
    return values, scale


def contraction_source(terms, cache, lines):
    """Factor shared operands in a sparse multilinear contraction."""
    terms = tuple(
        (coefficient, tuple(sorted(factors))) for coefficient, factors in terms
    )
    key = "contraction", terms
    if key in cache:
        return cache[key]
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
        rest = contraction_source(remaining, cache, lines) if remaining else "T(0)"
        expression = f"fma({factor}, {inner}, {rest})"
    else:
        products = []
        for coefficient, factors in terms:
            value = "T(1)"
            for operand in factors:
                if value == "T(1)":
                    value = operand
                else:
                    product = "angular_product", *sorted((value, operand))
                    if product not in cache:
                        name = f"v{len(cache)}"
                        lines.append(f"const T {name} = {value} * {operand};")
                        cache[product] = name
                    value = cache[product]
            products.append((coefficient, value))
        name = f"v{len(cache)}"
        cache[key] = name
        lines.append(f"T {name} = 0;")
        lines.extend(
            f"{name} = fma(T({coefficient:.17g}), {value}, {name});"
            for coefficient, value in products
        )
        return name
    name = f"v{len(cache)}"
    cache[key] = name
    lines.append(f"const T {name} = {expression};")
    return name
