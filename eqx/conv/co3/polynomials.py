"""Cartesian harmonic polynomials and sparse delta/epsilon contractions."""

import math
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import product

ANGULAR_TILE_SIZE = 27
FUSED_TILE_LIMIT = 16


def output_components(paths, irreps, normalization):
    """Identify identical output polynomials without merging coupling paths.

    Equality is checked separately for every incoming instruction and every
    input tensor entry. No symmetry of the input features is assumed.
    Degrees above four retain their full Cartesian storage.
    """
    groups = defaultdict(list)
    for path in paths:
        if path[6] <= 81:
            degree, dim = 0, path[6]
            while dim > 1:
                degree, dim = degree + 1, dim // 3
            polynomials = polynomial_coefficients(degree, normalization)
            representatives = {}
            coefficients = defaultdict(list)
            for a, b, c, value in path[-1]:
                b = representatives.setdefault(polynomials[b], b)
                coefficients[a, b, c].append(value)
            path = (
                *path[:-1],
                tuple(
                    (*indices, math.fsum(values))
                    for indices, values in sorted(coefficients.items())
                    if math.fsum(values)
                ),
            )
        groups[path[2]].append(path)
    output_paths, expansion = [], []
    start = offset = 0
    for mul, ir in irreps:
        if not mul:
            continue
        entries = groups[start]
        signatures = [[] for _ in range(ir.dim)]
        compact = ir.l <= 4 and all(p[6] <= 81 for p in entries)
        if compact:
            for path in entries:
                degree, dim = 0, path[6]
                while dim > 1:
                    degree, dim = degree + 1, dim // 3
                polynomials = polynomial_coefficients(degree, normalization)
                rows = [defaultdict(float) for _ in range(ir.dim)]
                for a, b, c, coefficient in path[-1]:
                    rows[c][a, polynomials[b]] += coefficient
                for signature, row in zip(signatures, rows):
                    signature.append(
                        tuple(
                            sorted((key, value) for key, value in row.items() if value)
                        )
                    )
        unique, representatives, indices = {}, [], []
        for c, signature in enumerate(signatures):
            key = tuple(signature) if compact and entries else c
            if key not in unique:
                unique[key] = len(unique)
                representatives.append(c)
            indices.append(unique[key])
        width = len(unique)
        columns = {c: i for i, c in enumerate(representatives)}
        for path in entries:
            coefficients = tuple(
                (a, b, columns[c], value) for a, b, c, value in path[-1] if c in columns
            )
            output_paths.append(
                (path[0], path[1], offset, *path[3:7], width, *path[8:10], coefficients)
            )
        expansion.extend(offset + c * mul + u for c in indices for u in range(mul))
        offset += mul * width
        start += mul * ir.dim
    return tuple(output_paths), tuple(expansion), offset


@lru_cache(maxsize=None)
def coupling_coefficients(l1, l2, l3):
    """Return sparse coefficients of an unprojected Cartesian coupling."""
    k, odd = divmod(l1 + l2 - l3, 2)
    left, contracted, right = 3 ** (l1 - k - odd), 3**k, 3 ** (l2 - k - odd)
    scale = 1 / math.sqrt(contracted * (2 if odd else 1))
    entries = []
    for i, j, t in product(range(left), range(right), range(contracted)):
        if odd:
            for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
                out = (i * 3 + c) * right + j
                entries.append(
                    ((i * 3 + a) * contracted + t, (t * 3 + b) * right + j, out, scale)
                )
                entries.append(
                    ((i * 3 + b) * contracted + t, (t * 3 + a) * right + j, out, -scale)
                )
        else:
            entries.append((i * contracted + t, t * right + j, i * right + j, scale))
    return tuple(entries)


@lru_cache(maxsize=None)
def polynomial_coefficients(degree, normalization):
    """Expand STF vector powers with exact rational trace subtraction.

    Coefficients are collected before applying the harmonic normalization.
    Tensor entries related by index permutations share a polynomial.
    """
    scale = math.sqrt(math.comb(2 * degree, degree) / 2**degree)
    if normalization == "component":
        scale *= math.sqrt(2 * degree + 1)
    elif normalization == "integral":
        scale *= math.sqrt((2 * degree + 1) / (4 * math.pi))
    elif normalization != "norm":
        raise ValueError("normalization must be integral, component or norm.")
    polynomials = {}
    result = []
    for indices in product(range(3), repeat=degree):
        counts = tuple(indices.count(a) for a in range(3))
        if counts not in polynomials:
            terms = defaultdict(Fraction)
            for pairs in product(*(range(n // 2 + 1) for n in counts)):
                k = sum(pairs)
                coefficient = Fraction(
                    (-1) ** k, math.prod(range(2 * degree - 2 * k + 1, 2 * degree, 2))
                )
                for n, t in zip(counts, pairs):
                    coefficient *= math.factorial(n) // (
                        math.factorial(n - 2 * t) * 2**t * math.factorial(t)
                    )
                for a in range(k + 1):
                    for b in range(k - a + 1):
                        powers = tuple(
                            n - 2 * t + 2 * p
                            for n, t, p in zip(counts, pairs, (a, b, k - a - b))
                        )
                        terms[powers] += (
                            coefficient * math.comb(k, a) * math.comb(k - a, b)
                        )
            polynomials[counts] = tuple(
                (powers, float(value) * scale)
                for powers, value in sorted(terms.items())
                if value
            )
        result.append(polynomials[counts])
    return tuple(result)
