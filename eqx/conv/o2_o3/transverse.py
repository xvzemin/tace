"""Spherical contractions of transverse irreducible tensors."""

import torch
from e3nn import o3

from ...co2.spherical import coupling_polynomial
from ...kernels.codegen import contraction_source
from ..angular import generator_action


def coupling_source(l1, l2, l3, vector, features, normalization, cache, lines):
    """Emit a transverse restriction and lift without Cartesian intermediates."""
    from ..o3.harmonics import polynomial_coefficients

    delta = abs(l1 - l3)
    coefficients = coupling_polynomial(l1, l2, l3, normalization)
    degree = min(l1, l3)
    spectral_scale = degree * (degree + 1) / max(1, degree**2)

    def bridge(features):
        bridge_key = "transverse_bridge", l1, l3, vector, features
        if bridge_key in cache:
            return cache[bridge_key]
        if l1 == l3:
            scale = (2 * l1 + 1) ** -0.5
            values = tuple(
                contraction_source(((scale, (value,)),), cache, lines)
                for value in features
            )
        else:
            cg = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")
            harmonics = tuple(
                contraction_source(
                    tuple(
                        (
                            c,
                            tuple(
                                vector[a]
                                for a, p in enumerate(powers)
                                for _ in range(p)
                            ),
                        )
                        for powers, c in polynomial
                    ),
                    cache,
                    lines,
                )
                for polynomial in polynomial_coefficients(delta, "norm")
            )
            entries = tuple(
                (a, b, c, float(cg[a, b, c])) for a, b, c in cg.nonzero().tolist()
            )
            values = tuple(
                contraction_source(
                    tuple(
                        (scale, (features[a], harmonics[b]))
                        for a, b, c, scale in entries
                        if c == out
                    ),
                    cache,
                    lines,
                )
                for out in range(2 * l3 + 1)
            )
        cache[bridge_key] = values
        return values

    values = features if l1 < l3 else bridge(features)
    if (l1 + l2 + l3) % 2:
        values = generator_action(degree, vector, values, cache, lines, normalized=True)
    terms = [[(coefficients[0], (value,))] for value in values]
    previous = values
    for k, coefficient in enumerate(coefficients[1:], start=1):
        squared = generator_action(
            degree,
            vector,
            generator_action(degree, vector, values, cache, lines, normalized=True),
            cache,
            lines,
            normalized=True,
        )
        following = tuple(
            contraction_source(
                ((-2.0 * spectral_scale, (a,)), (-1.0, (b,)))
                if k == 1
                else ((-4.0 * spectral_scale, (a,)), (-2.0, (b,)), (-1.0, (c,))),
                cache,
                lines,
            )
            for a, b, c in zip(squared, values, previous)
        )
        previous, values = values, following
        for row, value in zip(terms, values):
            row.append((coefficient, (value,)))
    result = tuple(contraction_source(tuple(row), cache, lines) for row in terms)
    return bridge(result) if l1 < l3 else result


def coupling_derivative_source(
    l1, l2, l3, vector, features, normalization, cache, lines, directions
):
    """Differentiate the generator polynomial by the exact product rule.

    Each subset of ``directions`` stores one mixed directional derivative.
    Generator actions remain sparse at every order; derivatives do not switch
    to a full harmonic CG contraction.
    """
    from ..o3.harmonics import polynomial_coefficients

    degree, delta = min(l1, l3), abs(l1 - l3)
    coefficients = coupling_polynomial(l1, l2, l3, normalization)
    spectral_scale = degree * (degree + 1) / max(1, degree**2)
    size = 1 << len(directions)
    subsets = tuple(
        tuple(s for s in range(size) if s & mask == s) for mask in range(size)
    )

    def contract(terms):
        return contraction_source(
            tuple((c, f) for c, f in terms if c and "T(0)" not in f), cache, lines
        )

    def generator(values):
        result = []
        for mask, value in enumerate(values):
            terms = [
                generator_action(degree, vector, value, cache, lines, normalized=True)
            ]
            terms.extend(
                generator_action(
                    degree,
                    direction,
                    values[mask ^ (1 << k)],
                    cache,
                    lines,
                    normalized=True,
                )
                for k, direction in enumerate(directions)
                if mask & (1 << k)
            )
            result.append(
                tuple(contract(tuple((1.0, (x,)) for x in row)) for row in zip(*terms))
            )
        return tuple(result)

    def bridge(values):
        if l1 == l3:
            scale = (2 * l1 + 1) ** -0.5
            return tuple(
                tuple(contract(((scale, (x,)),)) for x in row) for row in values
            )
        harmonics = []
        for polynomial in polynomial_coefficients(delta, "norm"):
            terms = [[] for _ in range(size)]
            for powers, coefficient in polynomial:
                jet = ("T(1)",) + ("T(0)",) * (size - 1)
                for a, power in enumerate(powers):
                    for _ in range(power):
                        jet = tuple(
                            contract(
                                ((1.0, (jet[mask], vector[a])),)
                                + tuple(
                                    (1.0, (jet[mask ^ (1 << k)], direction[a]))
                                    for k, direction in enumerate(directions)
                                    if mask & (1 << k)
                                )
                            )
                            for mask in range(size)
                        )
                for row, value in zip(terms, jet):
                    row.append((coefficient, (value,)))
            harmonics.append(tuple(contract(row) for row in terms))
        cg = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")
        entries = tuple(
            (a, b, c, float(cg[a, b, c])) for a, b, c in cg.nonzero().tolist()
        )
        return tuple(
            tuple(
                contract(
                    tuple(
                        (coefficient, (values[s][a], harmonics[b][mask ^ s]))
                        for a, b, c, coefficient in entries
                        if c == out
                        for s in subsets[mask]
                    )
                )
                for out in range(2 * l3 + 1)
            )
            for mask in range(size)
        )

    values = (features,) + (("T(0)",) * len(features),) * (size - 1)
    if l1 >= l3:
        values = bridge(values)
    if (l1 + l2 + l3) % 2:
        values = generator(values)
    terms = [[[(coefficients[0], (x,))] for x in row] for row in values]
    previous = values
    for k, coefficient in enumerate(coefficients[1:], start=1):
        squared = generator(generator(values))
        following = tuple(
            tuple(
                contract(
                    ((-2.0 * spectral_scale, (a,)), (-1.0, (b,)))
                    if k == 1
                    else ((-4.0 * spectral_scale, (a,)), (-2.0, (b,)), (-1.0, (c,)))
                )
                for a, b, c in zip(row_a, row_b, row_c)
            )
            for row_a, row_b, row_c in zip(squared, values, previous)
        )
        previous, values = values, following
        for rows, row in zip(terms, values):
            for entries, value in zip(rows, row):
                entries.append((coefficient, (value,)))
    result = tuple(tuple(contract(entries) for entries in row) for row in terms)
    return (bridge(result) if l1 < l3 else result)[-1]
