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
