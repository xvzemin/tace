"""Cartesian harmonic polynomials and sparse delta/epsilon contractions."""

import math
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import product

from ...co3.symmetric import symmetric_powers

ANGULAR_TILE_SIZE = 27
FUSED_TILE_LIMIT = 16


@lru_cache(maxsize=None)
def symmetric_path_matrix(degree):
    """Return spherical-to-unique-Cartesian coefficients in CPU float64."""
    import torch

    from ...co3.basis import path_matrix

    matrix = path_matrix(degree)
    lookup = {p: i for i, p in enumerate(symmetric_powers(degree))}
    indices = torch.tensor(
        [lookup[component_powers(i, degree)] for i in range(3**degree)],
        dtype=torch.long,
        device="cpu",
    )
    packed = matrix.new_zeros(len(lookup), 2 * degree + 1)
    return packed.index_add(0, indices, matrix) / indices.bincount()[:, None]


@lru_cache(maxsize=None)
def component_powers(index, degree):
    """Count x, y and z indices in a Cartesian component."""
    counts = [0, 0, 0]
    for _ in range(degree):
        counts[index % 3] += 1
        index //= 3
    return tuple(counts)


def symmetric_components(paths, irreps_in, irreps_out):
    """Generate delta/epsilon paths in partially symmetric storage.

    Input permutations share one coordinate. Output indices are interchangeable
    only within free-index groups shared by every incoming instruction.
    Contracted permutations contribute their exact integer multiplicities.
    """
    inputs, sources, targets = {}, [], []
    start = width = 0
    for mul, ir in irreps_in:
        if not mul:
            continue
        powers = symmetric_powers(ir.l)
        lookup = {p: i for i, p in enumerate(powers)}
        inputs[start] = width, ir.l, lookup
        for a in range(ir.dim):
            index = lookup[component_powers(a, ir.l)]
            for u in range(mul):
                sources.append(start + a * mul + u)
                targets.append(width + index * mul + u)
        start += mul * ir.dim
        width += mul * len(powers)
    incoming = defaultdict(list)
    for path in paths:
        incoming[path[2]].append(path)
    result, expansion = [], []
    start = offset = 0
    for mul, ir in irreps_out:
        if not mul:
            continue
        entries = incoming[start]
        boundaries = {0, ir.l}
        for path in entries:
            l1 = inputs[path[0]][1]
            l2, dim = 0, path[6]
            while dim > 1:
                l2, dim = l2 + 1, dim // 3
            k, odd = divmod(l1 + l2 - ir.l, 2)
            boundaries.update((l1 - k - odd, l1 - k))
        boundaries = sorted(boundaries)
        ranks = [b - a for a, b in zip(boundaries, boundaries[1:])]
        groups = tuple(product(*(symmetric_powers(rank) for rank in ranks)))
        lookup = {group: i for i, group in enumerate(groups)}
        representatives = [
            tuple(
                axis
                for p in group
                for axis, count in enumerate(p)
                for _ in range(count)
            )
            for group in groups
        ]
        for c in range(ir.dim):
            key = tuple(
                component_powers(c // 3 ** (ir.l - b) % 3 ** (b - a), b - a)
                for a, b in zip(boundaries, boundaries[1:])
            )
            expansion.extend(offset + lookup[key] * mul + u for u in range(mul))
        for path in entries:
            input_start, l1, input_lookup = inputs[path[0]]
            l2, dim = 0, path[6]
            while dim > 1:
                l2, dim = l2 + 1, dim // 3
            harmonic_lookup = {p: i for i, p in enumerate(symmetric_powers(l2))}
            k, odd = divmod(l1 + l2 - ir.l, 2)
            left = l1 - k - odd
            coefficients = defaultdict(int)
            for c, indices in enumerate(representatives):
                alpha = tuple(indices[:left].count(a) for a in range(3))
                beta = tuple(indices[left + odd :].count(a) for a in range(3))
                if odd:
                    a, b = ((1, 2), (2, 0), (0, 1))[indices[left]]
                    epsilon = ((a, b, 1), (b, a, -1))
                else:
                    epsilon = ((-1, -1, 1),)
                for gamma in symmetric_powers(k):
                    count = math.comb(k, gamma[0]) * math.comb(k - gamma[0], gamma[1])
                    for p, q, sign in epsilon:
                        a = tuple(
                            x + y + (axis == p)
                            for axis, (x, y) in enumerate(zip(alpha, gamma))
                        )
                        b = tuple(
                            x + y + (axis == q)
                            for axis, (x, y) in enumerate(zip(beta, gamma))
                        )
                        coefficients[input_lookup[a], harmonic_lookup[b], c] += (
                            sign * count
                        )
            result.append(
                (
                    input_start,
                    path[1],
                    offset,
                    *path[3:5],
                    len(input_lookup),
                    len(harmonic_lookup),
                    len(groups),
                    *path[8:10],
                    tuple(
                        (*key, value)
                        for key, value in sorted(coefficients.items())
                        if value
                    ),
                )
            )
        start += mul * ir.dim
        offset += mul * len(groups)
    return (
        tuple(result),
        tuple(sources),
        tuple(targets),
        width,
        tuple(expansion),
        offset,
    )


@lru_cache(maxsize=None)
def symmetric_indices(dim):
    """Group Cartesian entries related by permutations of tensor axes."""
    degree, width = 0, dim
    while width > 1:
        degree, width = degree + 1, width // 3
    groups = defaultdict(list)
    for a, indices in enumerate(product(range(3), repeat=degree)):
        groups[tuple(sorted(indices))].append(a)
    inverse = [None] * dim
    for indices in groups.values():
        for a in indices:
            inverse[a] = tuple(indices)
    return tuple(inverse)


def input_components(paths, input_dim):
    """Sum input entries with identical contraction coefficients on nodes.

    Harmonic indices must first be canonicalized by ``output_components``.
    Equivalence is checked over every output entry, so the map is exact
    even for nonsymmetric inputs.
    Independent instructions retain their weights and output offsets.
    """
    sources, targets, width = [], [], 0
    layouts, result = {}, []
    for path in paths:
        columns = defaultdict(list)
        for a, b, c, value in path[-1]:
            columns[a].append((b, c, value))
        groups = defaultdict(list)
        for a, column in sorted(columns.items()):
            groups[tuple(sorted(column))].append(a)
        compact = (
            path[5] <= 81
            and path[6] <= 81
            and any(len(group) > 1 for group in groups.values())
        )
        entries = (
            tuple(tuple(g) for g in groups.values())
            if compact
            else tuple((a,) for a in range(path[5]))
        )
        key = path[0], path[3], entries
        if key not in layouts:
            layouts[key] = width
            start, mul, _ = key
            for c, indices in enumerate(entries):
                for a in indices:
                    sources.extend(start + a * mul + u for u in range(mul))
                    targets.extend(width + c * mul + u for u in range(mul))
            width += len(entries) * mul
        coefficients = (
            tuple(
                (a, b, c, value)
                for a, column in enumerate(groups)
                for b, c, value in column
            )
            if compact
            else path[-1]
        )
        result.append(
            (layouts[key], *path[1:5], len(entries), *path[6:-1], coefficients)
        )
    if not paths:
        return paths, tuple(range(input_dim)), tuple(range(input_dim)), input_dim
    return tuple(result), tuple(sources), tuple(targets), width


def output_components(paths, irreps, normalization):
    """Identify identical output polynomials without merging coupling paths.

    Equality is checked separately for every incoming instruction and every
    input tensor entry.
    Output degrees above four retain their full Cartesian storage.
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
    """Return integer delta/epsilon coefficients, excluding path normalization."""
    k, odd = divmod(l1 + l2 - l3, 2)
    left, contracted, right = 3 ** (l1 - k - odd), 3**k, 3 ** (l2 - k - odd)
    entries = []
    for i, j, t in product(range(left), range(right), range(contracted)):
        if odd:
            for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
                out = (i * 3 + c) * right + j
                entries.append(
                    ((i * 3 + a) * contracted + t, (t * 3 + b) * right + j, out, 1)
                )
                entries.append(
                    ((i * 3 + b) * contracted + t, (t * 3 + a) * right + j, out, -1)
                )
        else:
            entries.append((i * contracted + t, t * right + j, i * right + j, 1))
    return tuple(entries)


@lru_cache(maxsize=None)
def harmonic_polynomial(counts, normalization):
    """Return one Cartesian harmonic polynomial with exact trace subtraction."""
    degree = sum(counts)
    scale = math.sqrt(math.comb(2 * degree, degree) / 2**degree)
    if normalization == "component":
        scale *= math.sqrt(2 * degree + 1)
    elif normalization == "integral":
        scale *= math.sqrt((2 * degree + 1) / (4 * math.pi))
    elif normalization != "norm":
        raise ValueError("normalization must be integral, component or norm.")
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
                terms[powers] += coefficient * math.comb(k, a) * math.comb(k - a, b)
    return tuple(
        (powers, float(value) * scale)
        for powers, value in sorted(terms.items())
        if value
    )


@lru_cache(maxsize=None)
def polynomial_coefficients(degree, normalization):
    """Return harmonic polynomials in full Cartesian index order."""
    return tuple(
        harmonic_polynomial(component_powers(index, degree), normalization)
        for index in range(3**degree)
    )
