"""Orthonormal path matrices and Cartesian coupling normalization."""

from functools import lru_cache
from math import factorial, ldexp, prod, sqrt

import torch
from e3nn import o3

from .irreps import Irreps


@lru_cache(maxsize=None)
def path_matrix(l):
    """Return the orthonormal path matrix.

    Parameters
    ----------
    l : int
        Non-negative angular degree.

    Returns
    -------
    torch.Tensor
        Cached CPU float64 matrix of shape (3**l, 2*l+1).
        Do not modify the returned matrix in place.
    """
    if l < 0:
        raise ValueError("The angular degree must be non-negative.")
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64, device="cpu")
    cg = o3.wigner_3j(1, l - 1, l, dtype=torch.float64, device="cpu")
    matrix = torch.einsum("an,bnc->bac", path_matrix(l - 1), cg).flatten(0, 1)
    return matrix / matrix.norm(dim=0)


@lru_cache(maxsize=None)
def path_normalization(l1, l2, l3):
    r"""Return the magnitude relating delta/epsilon contractions to unit-norm CG coefficients.

    Parameters
    ----------
    l1, l2, l3 : int
        Input and output degrees satisfying the triangle inequality.

    Returns
    -------
    float
        Positive Cartesian-to-CG scale, excluding the path phase.

    Notes
    -----
    For S = l1+l2+l3 and k = floor((l1+l2-l3)/2), the squared scale is
    ``(S+1)! product((S-2*l)!) / (3**k product((2*l)!))``.
    Integer arithmetic and binary rescaling avoid intermediate overflow.
    """
    if min(l1, l2, l3) < 0 or not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(
            "Degrees must be non-negative and satisfy the triangle inequality."
        )
    total = l1 + l2 + l3
    numerator = factorial(total + 1) * prod(
        factorial(total - 2 * l) for l in (l1, l2, l3)
    )
    denominator = 3 ** ((l1 + l2 - l3) // 2) * prod(
        factorial(2 * l) for l in (l1, l2, l3)
    )
    exponent = (numerator.bit_length() - denominator.bit_length()) // 2
    if exponent >= 0:
        denominator <<= 2 * exponent
    else:
        numerator <<= -2 * exponent
    return ldexp(sqrt(numerator / denominator), exponent)


class ChangeOfBasis(torch.nn.Module):
    """Convert spherical coordinates and Cartesian tensors using path matrices.

    Parameters
    ----------
    irreps : Irreps or str
        Multiplicities and representation labels, without regrouping.
    inverse : bool, optional
        False maps spherical coordinates to Cartesian tensors. True extracts
        spherical coordinates and discards non-STF components.
    """

    def __init__(self, irreps, inverse=False):
        super().__init__()
        cartesian = Irreps(irreps)
        spherical = cartesian.spherical()
        self.inverse = inverse
        self.irreps_in = cartesian if inverse else spherical
        self.irreps_out = spherical if inverse else cartesian
        self.slices_in = self.irreps_in.slices()
        for l in sorted({ir.l for _, ir in cartesian}):
            self.register_buffer(
                f"basis_{l}",
                path_matrix(l).to(torch.get_default_dtype()).clone(),
                persistent=False,
            )

    def forward(self, features):
        """Transform features of shape (..., irreps_in.dim) in mul_ir order."""
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        values = []
        for (mul, ir), section in zip(self.irreps_in, self.slices_in):
            matrix = getattr(self, f"basis_{ir.l}")
            x = features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
            values.append((x @ (matrix if self.inverse else matrix.T)).flatten(-2))
        return torch.cat(values, dim=-1) if values else features[..., :0]

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, inverse={self.inverse}"


class Projector(torch.nn.Module):
    """Project Cartesian tensors onto their symmetric traceless subspaces.

    Parameters
    ----------
    irreps : Irreps or str
        Representation labels and Cartesian storage layout.
    """

    def __init__(self, irreps):
        super().__init__()
        self.irreps_in = self.irreps_out = Irreps(irreps)
        self.basis = ChangeOfBasis(irreps, inverse=True)

    def forward(self, features):
        """Apply C C.T without constructing a square Cartesian projector."""
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        values = []
        for (mul, ir), section in zip(self.irreps_in, self.basis.slices_in):
            matrix = getattr(self.basis, f"basis_{ir.l}")
            x = features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
            values.append(((x @ matrix) @ matrix.T).flatten(-2))
        return torch.cat(values, dim=-1) if values else features[..., :0]

    def extra_repr(self):
        return str(self.irreps_in)
