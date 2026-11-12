"""Exact real quaternion polynomial coefficients shared by execution providers."""

import math
from collections import defaultdict
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
@torch.compiler.assume_constant_result
def polynomial_data(lmax):
    """Expand the symmetric-power representation in the real harmonic basis.

    Polynomial coefficients are accumulated as integers before applying the
    harmonic normalization. Structural zeros therefore need no threshold or
    numerical fit. The spin-half matrix is
    ``[[w - iy, x + iz], [-x + iz, w + iy]]``.
    """
    pointers, exponents, coefficients = [0], [], []
    for l in range(lmax + 1):
        n = 2 * l

        @lru_cache(maxsize=None)
        def complex_polynomial(row, column):
            result = defaultdict(int)
            for r in range(max(0, row - column), min(n - column, row) + 1):
                a, b, c, d = n - column - r, column - row + r, r, row - r
                wy, xz = [0] * (a + d + 1), [0] * (b + c + 1)
                for i in range(a + 1):
                    for j in range(d + 1):
                        wy[i + j] += (-1) ** i * math.comb(a, i) * math.comb(d, j)
                for i in range(b + 1):
                    for j in range(c + 1):
                        xz[i + j] += (-1) ** (c - j) * math.comb(b, i) * math.comb(c, j)
                scale = math.comb(n - column, r) * math.comb(column, row - r)
                for y, cy in enumerate(wy):
                    for z, cz in enumerate(xz):
                        if cy and cz:
                            result[a + d - y, b + c - z, y, z] += scale * cy * cz
            return result

        # Each entry is (complex index, power of i). The common degree phase
        # cancels between the real-to-complex matrix and its adjoint.
        basis = []
        for m in range(-l, l + 1):
            if m < 0:
                basis.append(((l + m, 3), (l - m, (1 - 2 * m) % 4)))
            elif m > 0:
                basis.append(((l - m, 0), (l + m, (2 * m) % 4)))
            else:
                basis.append(((l, 0),))
        for row in range(n + 1):
            for column in range(n + 1):
                real, imaginary = defaultdict(int), defaultdict(int)
                for i, pi in basis[row]:
                    for j, pj in basis[column]:
                        for powers, value in complex_polynomial(i, j).items():
                            phase = (powers[2] + powers[3] + pj - pi) % 4
                            (real if phase % 2 == 0 else imaginary)[powers] += (
                                value if phase < 2 else -value
                            )
                if any(imaginary.values()):
                    raise RuntimeError("The quaternion polynomial must be real.")
                scale = math.sqrt(math.comb(n, column) / math.comb(n, row))
                scale /= math.sqrt(len(basis[row]) * len(basis[column]))
                for powers, value in sorted(real.items()):
                    if value:
                        exponents.append(powers)
                        coefficients.append(value * scale)
                pointers.append(len(coefficients))
    return (
        tuple(pointers),
        tuple(exponents),
        tuple(coefficients),
    )


@lru_cache(maxsize=32)
def polynomial_coefficients(lmax):
    """Return CPU tensors for native coefficient tables."""
    pointers, exponents, coefficients = polynomial_data(lmax)
    return (
        torch.tensor(pointers, dtype=torch.int32, device="cpu"),
        torch.tensor(exponents, dtype=torch.int32, device="cpu"),
        torch.tensor(coefficients, dtype=torch.float64, device="cpu"),
    )
