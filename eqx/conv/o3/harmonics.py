"""Fixed Cartesian polynomials for spherical-harmonic convolutions."""

import math
from collections import defaultdict
from functools import lru_cache
from itertools import product


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
    path, mapping, dimensions, shared, roles, cache, lines, normalization
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
        row = {0: "source[edge]", 4: "target[edge]"}.get(role, "edge")
        if role not in (0, 4) and shared[pointer]:
            row = "0"
        return variable(
            (pointer, row, column, channel),
            f"active ? p{pointer}[({row}) * {dimensions[pointer]} + {column} + {channel}] : T(0)",
        )

    def multiply(a, b):
        return variable(("product", *sorted((a, b))), f"{a} * {b}")

    def harmonic(m, output=None, axis=None):
        if output is None:
            axis = None
        key = "harmonic", degree, m, mapping[5:], output, axis, normalization
        if key in cache:
            return cache[key]
        name = f"v{len(cache)}"
        cache[key] = name
        terms = []
        for axes in product(range(3), repeat=rank - (output is not None)):
            directions = iter(axes)
            orders, factors = [0, 0, 0], []
            for role in range(6, len(mapping)):
                a = axis if role == output else next(directions)
                orders[a] += 1
                if role != output:
                    factors.append(load(role, a))
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
            for factor in factors:
                value = multiply(value, factor)
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
                        for b in range(dim2):
                            inner_key = (
                                "harmonic_adjoint",
                                start,
                                end,
                                mul1,
                                mapping[0],
                                mapping[4],
                                cg,
                                b,
                            )
                            if inner_key not in cache:
                                products = [
                                    (
                                        coefficient,
                                        multiply(
                                            load(0, start + a * mul1, "u"),
                                            load(4, end + c * mul1, "u"),
                                        ),
                                    )
                                    for a, j, c, coefficient in cg
                                    if j == b
                                ]
                                inner = f"v{len(cache)}"
                                cache[inner_key] = inner
                                lines.append(f"T {inner} = 0;")
                                lines.extend(
                                    f"{inner} = fma(T({coefficient:.17g}), {value}, {inner});"
                                    for coefficient, value in products
                                )
                            y = harmonic(b, role if role >= 6 else None, column)
                            terms.append((1.0, multiply(cache[inner_key], y)))
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
                            terms.append((coefficient, multiply(y, other)))
                    name = f"v{len(cache)}"
                    cache[key] = name
                    lines.append(f"T {name} = 0;")
                    for coefficient, term in terms:
                        lines.append(
                            f"{name} = fma(T({coefficient:.17g}), {term}, {name});"
                        )
                    if role != 3:
                        amplitude = load(3, attr + v)
                        lines.append(f"{name} *= {amplitude};")
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
