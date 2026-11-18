"""Cartesian harmonics from the finite delta expansion."""

import math
from functools import lru_cache

import torch

from ...ops.torch.co3.cartesian_harmonics import (
    low_order,
    monomial_powers,
    polynomial_coefficients,
    scale,
)
from ...utils.autograd import forward_ad_active
from ..symmetric import permutation_classes


@lru_cache(maxsize=32)
def coefficient_matrix(degree, normalization):
    rows = polynomial_coefficients(degree)
    matrix = torch.zeros(len(rows), len(rows), dtype=torch.float64, device="cpu")
    factor = math.sqrt(math.comb(2 * degree, degree) / 2**degree) * scale(
        degree, normalization
    )
    for j, row in enumerate(rows):
        for i, coefficient in row:
            matrix[i, j] = float(coefficient) * factor
    return matrix


class Harmonics(torch.nn.Module):
    """Evaluate one Cartesian harmonic by symmetrized vector and delta products."""

    def __init__(self, degree, normalization, backend):
        super().__init__()
        self.degree, self.normalization, self.backend = degree, normalization, backend
        if degree > 2:
            self.powers = monomial_powers(degree)
            self.register_buffer(
                "matrix",
                coefficient_matrix(degree, normalization).clone(),
                persistent=False,
            )
            self.register_buffer(
                "indices", permutation_classes(degree)[0].clone(), persistent=False
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if self.degree > 2:
            self.matrix = (
                coefficient_matrix(self.degree, self.normalization)
                .to(self.matrix)
                .clone()
            )
        return self

    def forward(self, vectors):
        native = self.backend == "cuda" and vectors.is_cuda and not forward_ad_active()
        if self.degree <= 2:
            if native:
                from ...ops.cuda.co3.cartesian_harmonics import (
                    low_order as cuda_low_order,
                )

                return cuda_low_order(vectors, [self.degree], self.normalization)
            return low_order(vectors, (self.degree,), self.normalization)
        axes = []
        for axis in vectors.unbind(-1):
            powers = [torch.ones_like(axis), axis]
            for _ in range(2, self.degree + 1):
                powers.append(powers[-1] * axis)
            axes.append(powers)
        terms = []
        for powers in self.powers:
            factors = [axis[k] for axis, k in zip(axes, powers) if k]
            value = factors[0]
            for factor in factors[1:]:
                value = value * factor
            terms.append(value)
        monomials = torch.stack(terms, -1)
        matrix = self.matrix.to(vectors)
        if native:
            from ...ops.common.cuda.grid import linear

            result = linear(monomials, matrix.T)
        else:
            result = monomials @ matrix
        return result.index_select(-1, self.indices)
