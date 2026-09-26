"""Fixed Cartesian polynomials for spherical-harmonic convolutions."""

import math
from collections import defaultdict
from functools import lru_cache

from ...kernels.codegen import contraction_source
from ..angular import (
    generator_action,
    generator_adjoint,
    generator_source,
)


@lru_cache(maxsize=64)
def polynomial_coefficients(degree, normalization):
    """Expand regular solid harmonics without fitting or numerical thresholds.

    Rodrigues' formula is accumulated with integer coefficients, with the
    real harmonic normalization applied only after collecting monomials.
    The polar axis is y; positive and negative orders use the real and
    imaginary parts of ``(z + i*x)**m``, respectively.
    """
    result = []
    for order in range(-degree, degree + 1):
        m = abs(order)
        terms = defaultdict(int)
        for k in range((degree - m) // 2 + 1):
            coefficient = (
                (-1) ** k
                * math.factorial(2 * degree - 2 * k)
                // (
                    math.factorial(k)
                    * math.factorial(degree - k)
                    * math.factorial(degree - 2 * k - m)
                )
            )
            for x in range(order < 0, m + 1, 2):
                angular = math.comb(m, x) * (-1) ** (x // 2)
                for a in range(k + 1):
                    for b in range(k - a + 1):
                        c = k - a - b
                        powers = (x + 2 * a, degree - 2 * k - m + 2 * b, m - x + 2 * c)
                        terms[powers] += (
                            coefficient
                            * angular
                            * math.comb(k, a)
                            * math.comb(k - a, b)
                        )
        scale = (
            math.sqrt(
                (2 * degree + 1)
                * (2 if m else 1)
                * math.factorial(degree - m)
                / math.factorial(degree + m)
            )
            / 2**degree
        )
        if normalization == "integral":
            scale /= math.sqrt(4 * math.pi)
        elif normalization == "norm":
            scale /= math.sqrt(2 * degree + 1)
        elif normalization != "component":
            raise ValueError("normalization must be integral, component or norm.")
        result.append(
            tuple(
                (powers, value * scale)
                for powers, value in sorted(terms.items())
                if value
            )
        )
    return tuple(result)


def angular_source(
    path,
    mapping,
    dimensions,
    shared,
    roles,
    cache,
    lines,
    normalization,
    use_generators=False,
    angular_derivatives=False,
    direction_offset=0,
):
    """Contract harmonic derivatives before reducing feature channels."""
    start, attr, end, mul1, mul2, dim1, dim2, dim_out, _, _, cg = path
    degree = (dim2 - 1) // 2
    rank = len(mapping) - 6

    def variable(key, expression):
        if key not in cache:
            name = f"v{len(cache)}"
            lines.append(f"const T {name} = {expression};")
            cache[key] = name
        return cache[key]

    def load(role, column, channel="0"):
        pointer = mapping[role]
        if role == 5:
            # The aligned contraction already stores the direction in row y
            # of the degree-one Wigner matrix. Read it without an edge copy.
            column += direction_offset
        row = {0: "source[edge]", 4: "target[edge]"}.get(role, "edge")
        if role not in (0, 4) and shared[pointer]:
            row = "0"
        return variable(
            (pointer, row, column, channel),
            f"active ? p{pointer}[({row}) * {dimensions[pointer]} + {column} + {channel}] : T(0)",
        )

    def multiply(a, b):
        return variable(("product", *sorted((a, b))), f"{a} * {b}")

    def angular_harmonics(start=6):
        key = "angular_harmonics", degree, mapping[5], mapping[start:], normalization
        if key not in cache:
            if start < len(mapping):
                vector = tuple(load(start, a) for a in range(3))
                values = generator_action(
                    degree, vector, angular_harmonics(start + 1), cache, lines
                )
            else:
                values = tuple(
                    contraction_source(
                        tuple(
                            (
                                coefficient,
                                tuple(
                                    load(5, a)
                                    for a, p in enumerate(powers)
                                    for _ in range(p)
                                ),
                            )
                            for powers, coefficient in polynomial
                        ),
                        cache,
                        lines,
                    )
                    for polynomial in polynomial_coefficients(degree, normalization)
                )
            cache[key] = values
        return cache[key]

    def harmonic_adjoint():
        values = []
        for b in range(dim2):
            key = "harmonic_adjoint", start, end, mul1, mapping[0], mapping[4], cg, b
            if key not in cache:
                cache[key] = contraction_source(
                    tuple(
                        (
                            coefficient,
                            (
                                load(0, start + a * mul1, "u"),
                                load(4, end + c * mul1, "u"),
                            ),
                        )
                        for a, j, c, coefficient in cg
                        if j == b
                    ),
                    cache,
                    lines,
                )
            values.append(cache[key])
        return tuple(values)

    def harmonic(m, output=None, axis=None):
        if angular_derivatives:
            return angular_harmonics()[m]
        if output is None:
            axis = None
        key = "harmonic", degree, m, mapping[5:], output, axis, normalization
        if key in cache:
            return cache[key]
        name = f"v{len(cache)}"
        cache[key] = name
        terms = []
        directions = {(0, 0, 0): "T(1)"}
        for role in range(6, len(mapping)):
            if role == output:
                continue
            updated = {}
            for orders, value in directions.items():
                for a in range(3):
                    powers = tuple(n + (i == a) for i, n in enumerate(orders))
                    factor = load(role, a)
                    term = factor if value == "T(1)" else multiply(value, factor)
                    updated.setdefault(powers, []).append(term)
            directions = {
                orders: values[0]
                if len(values) == 1
                else variable(
                    ("symmetric_directions", tuple(values)), " + ".join(values)
                )
                for orders, values in updated.items()
            }
        for orders, direction_factor in directions.items():
            if output is not None:
                orders = tuple(n + (i == axis) for i, n in enumerate(orders))
            partial_key = "partial", degree, m, mapping[5], tuple(orders), normalization
            if partial_key not in cache:
                polynomial = []
                for powers, coefficient in polynomial_coefficients(
                    degree, normalization
                )[m]:
                    if any(n > p for n, p in zip(orders, powers)):
                        continue
                    monomial = []
                    for a, (p, n) in enumerate(zip(powers, orders)):
                        coefficient *= math.factorial(p) // math.factorial(p - n)
                        monomial.extend([load(5, a)] * (p - n))
                    value = "T(1)"
                    for factor in monomial:
                        value = factor if value == "T(1)" else multiply(value, factor)
                    polynomial.append((coefficient, value))
                partial = f"v{len(cache)}"
                cache[partial_key] = partial
                lines.append(f"T {partial} = 0;")
                for coefficient, value in polynomial:
                    lines.append(
                        f"{partial} = fma(T({coefficient:.17g}), {value}, {partial});"
                    )
            value = cache[partial_key]
            if direction_factor != "T(1)":
                value = multiply(value, direction_factor)
            terms.append(value)
        lines.append(f"T {name} = 0;")
        lines.extend(f"{name} += {value};" for value in terms)
        return name

    needed = set(roles) - {1, 2, 5}
    weight_role = None
    if 1 in roles:
        widths = {0: dim1, 3: 1, 4: dim_out}
        weight_role = min(needed & widths.keys() or widths, key=widths.__getitem__)
        needed.add(weight_role)
    values = {}
    for v in range(mul2):
        for role in sorted(needed):
            width = {0: dim1, 3: 1, 4: dim_out}.get(role, 3)
            if angular_derivatives and role >= 6:
                # Transpose the prefix and reuse the suffix. This produces all
                # three vector components without three generator chains.
                cotangent = harmonic_adjoint()
                for index in range(6, role):
                    vector = tuple(load(index, a) for a in range(3))
                    cotangent = generator_action(
                        degree, vector, cotangent, cache, lines
                    )
                adjoint = generator_adjoint(
                    degree, angular_harmonics(role + 1), cotangent, cache, lines
                )
                amplitude = load(3, attr + v)
                sign = -1 if (role - 6) % 2 else 1
                for column, value in enumerate(adjoint):
                    values[v, role, column] = variable(
                        ("angular_adjoint", value, amplitude, sign),
                        f"T({sign}) * {value} * {amplitude}",
                    )
                continue
            if (
                use_generators
                and rank == 0
                and dim1 == dim_out
                and degree in (1, 2)
                and role in (0, 4)
            ):
                other = 4 if role == 0 else 0
                offset = end if role == 0 else start
                features = tuple(
                    load(other, offset + a * mul1, "u") for a in range(dim1)
                )
                vector = tuple(load(5, a) for a in range(3))
                coupled, scale = generator_source(
                    (dim1 - 1) // 2,
                    degree,
                    vector,
                    features,
                    cache,
                    lines,
                    normalization,
                )
                if role == 0 and degree == 1:
                    scale = -scale
                amplitude = load(3, attr + v)
                for column, value in enumerate(coupled):
                    values[v, role, column] = variable(
                        ("scaled_generator", value, amplitude, scale),
                        f"T({scale:.17g}) * {value} * {amplitude}",
                    )
                continue
            for column in range(width):
                key = (
                    "angular",
                    path,
                    tuple((i, p) for i, p in enumerate(mapping) if i != role),
                    role,
                    column,
                    v,
                )
                if key not in cache:
                    terms = []
                    if role == 3 or role >= 6:
                        # The CG adjoint is shared by all three Cartesian
                        # directions and stays in registers until contraction.
                        for b, value in enumerate(harmonic_adjoint()):
                            y = harmonic(b, role if role >= 6 else None, column)
                            terms.append((1.0, (value, y)))
                    else:
                        for a, b, c, coefficient in cg:
                            if (role == 0 and a != column) or (
                                role == 4 and c != column
                            ):
                                continue
                            y = harmonic(b)
                            other = (
                                load(4, end + c * mul1, "u")
                                if role == 0
                                else load(0, start + a * mul1, "u")
                            )
                            terms.append((coefficient, (y, other)))
                    value = contraction_source(terms, cache, lines)
                    name = f"v{len(cache)}"
                    cache[key] = name
                    if role != 3:
                        amplitude = load(3, attr + v)
                        lines.append(f"const T {name} = {value} * {amplitude};")
                    else:
                        lines.append(f"const T {name} = {value};")
                values[v, role, column] = cache[key]
        if weight_role is not None:
            role = weight_role
            offset, width, channels, channel = {
                0: (start, dim1, mul1, "u"),
                3: (attr + v, 1, 1, "0"),
                4: (end, dim_out, mul1, "u"),
            }[role]
            terms = tuple(
                (values[v, role, m], load(role, offset + m * channels, channel))
                for m in range(width)
            )
            key = "weight", terms
            if key not in cache:
                name = f"v{len(cache)}"
                cache[key] = name
                lines.append(f"T {name} = 0;")
                lines.extend(f"{name} = fma({a}, {b}, {name});" for a, b in terms)
            values[v, 1, 0] = cache[key]
    return values
