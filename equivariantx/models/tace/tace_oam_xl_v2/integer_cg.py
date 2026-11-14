"""Factor real spherical-basis normalization out of sparse CG expressions.

This is an experimental code-generation transformation, not a parameter or
checkpoint transformation. Integer coefficients describe the same analytical
CG tensor; the input metadata may already have rounded its constants to FP32.
The reconstruction error is checked before emitting a native program.
"""

from fractions import Fraction
from functools import lru_cache, reduce
import math


@lru_cache(maxsize=128)
def integer_path(path):
    degrees = [(dim - 1) // 2 for dim in path[5:8]]
    scales = [
        tuple(
            math.sqrt(math.factorial(l + m) * math.factorial(l - m) / (2 if m else 1))
            for m in range(-l, l + 1)
        )
        for l in degrees
    ]
    normalized = [
        c / (scales[0][a] * scales[1][b] * scales[2][z]) for a, b, z, c in path[-1]
    ]
    fractions = [Fraction(v / normalized[0]).limit_denominator(512) for v in normalized]
    denominator = reduce(math.lcm, [v.denominator for v in fractions], 1)
    coefficients = [v.numerator * (denominator // v.denominator) for v in fractions]
    divisor = reduce(math.gcd, coefficients)
    coefficients = [v // divisor for v in coefficients]
    gamma = sum(v * n for v, n in zip(normalized, coefficients)) / sum(
        n * n for n in coefficients
    )
    relative = max(abs(gamma * n / v - 1) for v, n in zip(normalized, coefficients))
    if relative > 2e-7 or max(map(abs, coefficients)) > 4096:
        raise ValueError(
            f"CG tensor does not match bounded integer normalization: {relative}"
        )
    cg = tuple((a, b, z, float(n)) for (a, b, z, _), n in zip(path[-1], coefficients))
    return (*path[:-1], cg), tuple(scales), gamma, relative


def scaled_angular_source(
    base, path, mapping, dimensions, shared, roles, cache, lines, premultiply=None
):
    if premultiply is not None:
        raise ValueError(
            "Integer normalization and folded inputs are separate experiments"
        )
    converted, scales, gamma, _ = integer_path(path)
    required = set()
    for role in roles:
        if role in (0, 3, 4):
            required.update({0, 3, 4} - {role})
        elif role == 1:
            required.update((0, 3, 4))
    for role, offset, width, dim, scale in (
        (0, path[0], path[3], path[5], scales[0]),
        (3, path[1], path[4], path[6], scales[1]),
        (4, path[2], path[3], path[7], scales[2]),
    ):
        if role not in required:
            continue
        pointer = mapping[role]
        row = {
            0: "source_node",
            3: "0" if shared[pointer] else "edge",
            4: "target_node",
        }[role]
        for m in range(dim):
            for v in range(path[4]) if role == 3 else (0,):
                channel = str(v) if role == 3 else "u"
                column = offset + m * width
                key = pointer, row, column, channel
                if key in cache:
                    continue
                name = f"v{len(cache)}"
                lines.append(
                    f"const T {name}=active ? p{pointer}[({row})*{dimensions[pointer]}+{column}+{channel}]*T({scale[m]:.17g}) : T(0);"
                )
                cache[key] = name
    values = base(converted, mapping, dimensions, shared, roles, cache, lines)
    result = {}
    for (v, role, m), value in values.items():
        factor = gamma * (scales[{0: 0, 3: 1, 4: 2}[role]][m] if role != 1 else 1.0)
        key = "integer_cg_output", value, factor
        if key not in cache:
            name = f"v{len(cache)}"
            cache[key] = name
            lines.append(f"const T {name}={value}*T({factor:.17g});")
        result[v, role, m] = cache[key]
    return result
