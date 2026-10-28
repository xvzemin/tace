"""Irreducible Cartesian products with analytic trace subtraction."""

import math
from fractions import Fraction
from functools import lru_cache

import torch

from ..ictp_ictc.tensor_product import TensorProduct as ContractionTensorProduct
from ..ictp_ictc.tensor_product import cartesian_product
from ..symmetric import SymmetricBasis, symmetric_basis, symmetric_powers

SYMMETRIC_MATRIX_SIZE = 8192


@lru_cache(maxsize=None)
def identity_expansion(rank, traces):
    """Return the normalized symmetric map for inserting Kronecker deltas."""
    lower = rank - 2 * traces
    powers = symmetric_powers(rank)
    lookup = {p: i for i, p in enumerate(powers)}
    matrix = torch.zeros(
        len(symmetric_powers(lower)), len(powers), dtype=torch.float64, device="cpu"
    )
    for i, q in enumerate(symmetric_powers(lower)):
        multiplicity_in = math.comb(lower, q[0]) * math.comb(lower - q[0], q[1])
        for b in symmetric_powers(traces):
            p = tuple(v + 2 * w for v, w in zip(q, b))
            multiplicity_out = math.comb(rank, p[0]) * math.comb(rank - p[0], p[1])
            multiplicity_delta = math.comb(traces, b[0]) * math.comb(
                traces - b[0], b[1]
            )
            matrix[i, lookup[p]] = (
                math.sqrt(Fraction(multiplicity_in, multiplicity_out))
                * multiplicity_delta
            )
    return matrix


@lru_cache(maxsize=None)
def symmetric_expansion(rank, traces):
    """Combine symmetrization and delta insertion in unique Cartesian storage."""
    lower = rank - 2 * traces
    data = symmetric_basis(lower)
    matrix = torch.zeros(
        3**lower, len(data["norm"]), dtype=torch.float64, device="cpu"
    ).scatter_(
        1,
        data["indices"].unsqueeze(-1),
        data["norm"][data["indices"]].reciprocal().unsqueeze(-1),
    )
    return (matrix @ identity_expansion(rank, traces)) / symmetric_basis(rank)["norm"]


class CartesianProduct(torch.nn.Module):
    """Couple one pair of STF tensors by symmetric delta and epsilon products.

    Parameters
    ----------
    l1, l2, l3 : int
        Input and output ranks satisfying the triangle inequality.
    """

    def __init__(self, l1, l2, l3):
        super().__init__()
        self.l1, self.l2, self.l3 = l1, l2, l3
        k, odd = divmod(l1 + l2 - l3, 2)
        a, b = l1 - k - odd, l2 - k - odd
        self.symmetric = l3 >= 2 and l3 != abs(l1 - l2)
        if not self.symmetric:
            return
        self.packed = 3**l3 * (l3 + 1) * (l3 + 2) // 2 <= SYMMETRIC_MATRIX_SIZE
        if self.packed:
            self.register_buffer(
                "indices", symmetric_basis(l3)["indices"].clone(), persistent=False
            )
        else:
            self.bases = torch.nn.ModuleList(
                SymmetricBasis(l3 - 2 * t) for t in range(min(a, b) + 1)
            )
        coefficient = Fraction(1)
        self.coefficients = [1.0]
        for t in range(1, min(a, b) + 1):
            coefficient *= Fraction(
                -(a - t + 1) * (b - t + 1), t * (2 * l3 - 2 * t + 1)
            )
            # cartesian_product normalizes each additional contraction by sqrt(3).
            scale = float(coefficient) * math.sqrt(3**t)
            self.coefficients.append(scale)
        for t, scale in enumerate(self.coefficients):
            if not t and not self.packed:
                continue
            matrix = (
                symmetric_expansion(l3, t) if self.packed else identity_expansion(l3, t)
            )
            self.register_buffer(
                f"expansion_{t}",
                (matrix * scale).to(torch.get_default_dtype()),
                persistent=False,
            )

    def forward(self, x, y):
        """Return the STF coupling in flattened Cartesian storage."""
        value = cartesian_product(x, y, self.l1, self.l2, self.l3)
        if not self.symmetric:
            return value
        if self.packed:
            result = value @ self.expansion_0
            for t in range(1, len(self.coefficients)):
                value = cartesian_product(x, y, self.l1, self.l2, self.l3 - 2 * t)
                result = result + value @ getattr(self, f"expansion_{t}")
            return result.index_select(-1, self.indices)
        result = self.bases[0].pack(value)
        for t, basis in enumerate(self.bases):
            if t:
                value = basis.pack(
                    cartesian_product(x, y, self.l1, self.l2, self.l3 - 2 * t)
                )
                result = result + value @ getattr(self, f"expansion_{t}")
        return self.bases[0].unpack(result)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if not self.symmetric:
            return self
        for t, scale in enumerate(self.coefficients):
            if not t and not self.packed:
                continue
            name = f"expansion_{t}"
            matrix = (
                symmetric_expansion(self.l3, t)
                if self.packed
                else identity_expansion(self.l3, t)
            )
            setattr(
                self,
                name,
                (matrix * scale).to(getattr(self, name)),
            )
        return self


class TensorProduct(ContractionTensorProduct):
    """Couple Cartesian irreps with analytic symmetrization and trace removal.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Input and output representations in flattened ``mul_ir`` layout.
    instructions : sequence of tuple
        ``(i_in1, i_in2, i_out, mode, has_weight[, path_weight])`` entries.
        Modes are ``uvu``, ``uvv``, ``uuu``, ``uuw``, ``uvw``, and ``uvuv``.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of independent irreducible coordinates.
    path_normalization : {"element", "path", "none"}, optional
        Normalization of paths feeding the same output entry.
    in1_var, in2_var, out_var : sequence of float, optional
        Variance for each input or output entry. Defaults to one.
    internal_weights, shared_weights : bool, optional
        Store trainable weights and share weights over leading dimensions.
        Internal weights require shared weights.

    Notes
    -----
    Inputs and outputs are symmetric traceless tensors. Angular paths and
    weights use the same normalization as :class:`eqx.co3.TensorProduct`.
    Delta and Levi-Civita contractions support both spatial parities.

    Examples
    --------
    >>> from eqx import co3
    >>> tp = co3.ictp.TensorProduct(
    ...     "2x1o", "2x1o", "2x2e", [(0, 0, 0, "uuu", True)]
    ... )
    >>> tp(torch.randn(5, 6), torch.randn(5, 6)).shape
    torch.Size([5, 18])
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        *,
        irrep_normalization="component",
        path_normalization="element",
        in1_var=None,
        in2_var=None,
        out_var=None,
        internal_weights=None,
        shared_weights=None,
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            project=False,
        )
        self.project = True
        self.products = torch.nn.ModuleDict()
        for ins in self.instructions:
            degrees = (
                self.irreps_in1[ins.i_in1].ir.l,
                self.irreps_in2[ins.i_in2].ir.l,
                self.irreps_out[ins.i_out].ir.l,
            )
            key = "_".join(map(str, degrees))
            if key not in self.products:
                self.products[key] = CartesianProduct(*degrees)

    def couple(self, x, y, l1, l2, l3):
        """Evaluate one irreducible angular coupling."""
        return self.products[f"{l1}_{l2}_{l3}"](x, y)
