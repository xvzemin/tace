"""Gaunt tensor products of spherical features."""

import math

import torch
from e3nn import o3

from ..s2grid import S2Grid
from ..tensor_product import FactorizedTensorProduct

__all__ = ["GauntTensorProduct"]


class GauntTensorProduct(FactorizedTensorProduct):
    """Multiply spherical signals with both spatial parities.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : e3nn.o3.Irreps or str
        Input and output representations. Each has uniform multiplicity after
        grouping equal irreps. Inputs match the output multiplicity or have
        multiplicity one for channel broadcasting. Spatial and time parities
        follow the supplied e3nn representation labels.
    irrep_normalization : {"component", "norm", "none"}, optional
        Unit component variance, unit expected squared irrep norm, or the
        unscaled integral of three surface-orthonormal spherical harmonics.
        Variance normalization assumes independent inputs and unit-variance
        factors. Defaults to ``"component"``.
    internal_weights, shared_weights : bool, optional
        Store trainable factors and share them over leading dimensions.
        Sharing defaults to true; internal weights default to shared_weights.
    method : {"quadrature", "coefficients"}, optional
        Exact-degree sphere quadrature or direct coefficient contraction.
        Defaults to ``"quadrature"``.

    Notes
    -----
    Features use e3nn's real basis and ``mul_ir`` layout. Weight factors act
    on each input irrep/channel and each output irrep/channel/sector pair,
    in that order. They do not parameterize independent CG paths. Pass ones
    for the unweighted product. Output components without a coupling are zero.
    Gaunt couplings with odd ``l1 + l2 + l_out`` vanish, including
    ``1o x 1o -> 1e``. Both natural and unnatural spatial parities are supported.

    Examples
    --------
    >>> from eqx import o3
    >>> tp = o3.GauntTensorProduct("2x1o", "2x1o", "2x0e + 2x2e")
    >>> x = torch.randn(5, tp.irreps_in1.dim)
    >>> y = torch.randn(5, tp.irreps_in2.dim)
    >>> tp(x, y).shape
    torch.Size([5, 12])
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
        method="quadrature",
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization,
            internal_weights,
            shared_weights,
        )
        if method not in ("quadrature", "coefficients"):
            raise ValueError("method must be quadrature or coefficients.")
        self.method = method
        lmax = max(
            irreps.lmax
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        degree = sum(
            irreps.lmax
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        grid = S2Grid(
            lmax,
            max(1, 2 * lmax, degree),
            normalization="integral",
            quadrature="gauss_legendre",
            dtype=torch.float64,
            device="cpu",
        )
        self.grid_basis = grid.synthesis
        self.grid_analysis = grid.analysis
        self.build()
        del self.grid_basis, self.grid_analysis

    @staticmethod
    def sector(ir):
        return ir.p * (-1) ** ir.l

    @staticmethod
    def reduced_coefficient(l1, l2, l_out):
        if (l1 + l2 + l_out) % 2:
            return 0.0
        cg = o3.wigner_3j(l1, l2, l_out, dtype=torch.float64, device="cpu")
        return float(cg[l1, l2, l_out]) * math.sqrt(
            (2 * l1 + 1) * (2 * l2 + 1) * (2 * l_out + 1) / (4 * math.pi)
        )

    def input_basis(self, irreps):
        return torch.cat(
            [self.grid_basis[:, ir.l**2 : (ir.l + 1) ** 2] for ir in irreps], -1
        ).T

    def output_basis(self, irreps):
        return torch.cat(
            [self.grid_analysis[ir.l**2 : (ir.l + 1) ** 2] for ir in irreps], 0
        ).T

    def coupling_tensor(self, irreps1, irreps2, irreps_out):
        coefficients = torch.zeros(
            sum(ir.dim for ir in irreps1),
            sum(ir.dim for ir in irreps2),
            sum(ir.dim for ir in irreps_out),
            dtype=torch.float64,
            device="cpu",
        )
        a = 0
        for ir1 in irreps1:
            b = 0
            for ir2 in irreps2:
                c = 0
                for ir in irreps_out:
                    if (
                        ir.p == ir1.p * ir2.p
                        and getattr(ir, "t", 1)
                        == getattr(ir1, "t", 1) * getattr(ir2, "t", 1)
                        and abs(ir1.l - ir2.l) <= ir.l <= ir1.l + ir2.l
                    ):
                        coefficients[
                            a : a + ir1.dim, b : b + ir2.dim, c : c + ir.dim
                        ] = self.reduced_coefficient(ir1.l, ir2.l, ir.l) * o3.wigner_3j(
                            ir1.l, ir2.l, ir.l, dtype=torch.float64, device="cpu"
                        )
                    c += ir.dim
                b += ir2.dim
            a += ir1.dim
        return coefficients

    def transform(self, features, basis):
        return features if self.method == "coefficients" else features @ basis

    def product(self, x, y, name):
        if self.method == "coefficients":
            return torch.einsum(
                "...ci,...cj,ijk->...ck", x, y, getattr(self, f"{name}_coupling")
            )
        return (x * y) @ getattr(self, f"{name}_basis")
