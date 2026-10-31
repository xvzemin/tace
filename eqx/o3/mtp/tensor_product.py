"""Matrix tensor products of spherical features."""

import math

import torch
from e3nn import o3

from ..tensor_product import FactorizedTensorProduct

__all__ = ["MatrixTensorProduct"]


class MatrixTensorProduct(FactorizedTensorProduct):
    """Multiply matrix representations of spherical features with spatial parity.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : e3nn.o3.Irreps or str
        Representations with uniform multiplicity after grouping equal irreps.
        Inputs match the output multiplicity or have multiplicity one.
        Spatial and time parities follow the supplied e3nn representation labels.
    irrep_normalization : {"component", "norm", "none"}, optional
        Unit component variance, unit expected squared irrep norm, or unscaled
        multiplication in the orthonormal matrix basis. Normalization assumes
        independent inputs and unit-variance factors. Defaults to ``"component"``.
    internal_weights, shared_weights : bool, optional
        Store trainable factors and share them over leading dimensions.
        Sharing defaults to true; internal weights default to shared_weights.
    matrix_degree : int, optional
        Auxiliary angular degree. Matrices have size ``2 * matrix_degree + 1``.
        Defaults to half the largest requested degree, rounded up.

    Notes
    -----
    Features use e3nn's real basis and ``mul_ir`` layout. Factors act on each
    input irrep/channel and each output irrep/channel/parity pair, in that
    order. Pass ones for unweighted matrix multiplication. Relative CG path
    coefficients are fixed by matrix_degree; individual paths can vanish.
    Changing matrix_degree changes the operator. Both spatial parities and
    antisymmetric couplings are supported. Channels are not mixed.

    Examples
    --------
    >>> from eqx import o3
    >>> tp = o3.MatrixTensorProduct("2x1o", "2x1o", "2x0e + 2x1e + 2x2e")
    >>> x = torch.randn(5, tp.irreps_in1.dim)
    >>> y = torch.randn(5, tp.irreps_in2.dim)
    >>> tp(x, y).shape
    torch.Size([5, 18])
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        *,
        irrep_normalization="component",
        internal_weights=None,
        shared_weights=None,
        matrix_degree=None,
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization,
            internal_weights,
            shared_weights,
        )
        lmax = max(
            irreps.lmax
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        matrix_degree = (lmax + 1) // 2 if matrix_degree is None else matrix_degree
        if not isinstance(matrix_degree, int) or 2 * matrix_degree < lmax:
            raise ValueError(
                "matrix_degree must be an integer with 2 * matrix_degree >= lmax."
            )
        self.matrix_degree = matrix_degree
        self.matrix_dim = 2 * matrix_degree + 1
        self.matrix_basis = {
            l: math.sqrt(2 * l + 1)
            * o3.wigner_3j(
                matrix_degree, matrix_degree, l, dtype=torch.float64, device="cpu"
            )
            for l in {ir.l for _, layout in self.layouts for ir in layout}
        }
        self.build()
        del self.matrix_basis

    @staticmethod
    def sector(ir):
        return ir.p

    def reduced_coefficient(self, l1, l2, l_out):
        cg = o3.wigner_3j(l1, l2, l_out, dtype=torch.float64, device="cpu")
        index = torch.unravel_index(cg.abs().argmax(), cg.shape)
        a, b, c = (int(item) for item in index)
        value = (
            (self.matrix_basis[l1][..., a] @ self.matrix_basis[l2][..., b])
            * self.matrix_basis[l_out][..., c]
        ).sum()
        coefficient = float(value / cg[a, b, c])
        return (
            0.0
            if abs(coefficient) < 64 * torch.finfo(torch.float64).eps
            else coefficient
        )

    def input_basis(self, irreps):
        return torch.cat([self.matrix_basis[ir.l] for ir in irreps], -1).flatten(0, 1).T

    def output_basis(self, irreps):
        return self.input_basis(irreps).T

    def transform(self, features, basis):
        return (features @ basis).reshape(
            features.shape[:-1] + (self.matrix_dim, self.matrix_dim)
        )

    def product(self, x, y, name):
        return (x @ y).flatten(-2) @ getattr(self, f"{name}_basis")
