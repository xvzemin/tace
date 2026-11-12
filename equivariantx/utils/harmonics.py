"""Exact Cartesian coefficients of regular solid harmonics."""

import math
from collections import defaultdict
from functools import lru_cache


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


def harmonic_program(program, direction, degree, normalization="norm"):
    """Encode solid harmonics with nonsingular polynomial derivatives."""
    powers = {(axis, 0): program.constant(1, 1) for axis in range(3)}
    for axis in range(3):
        powers[axis, 1] = program.slice(direction, axis, 1)
        for exponent in range(2, degree + 1):
            powers[axis, exponent] = program.binary(
                "mul", powers[axis, exponent - 1], powers[axis, 1]
            )
    values = []
    for component in polynomial_coefficients(degree, normalization):
        terms = []
        for monomial, coefficient in component:
            term = program.constant(1, coefficient)
            for axis, exponent in enumerate(monomial):
                if exponent:
                    term = program.binary("mul", term, powers[axis, exponent])
            terms.append(term)
        values.append(program.sum(terms, 1))
    return program.concatenate(values)
