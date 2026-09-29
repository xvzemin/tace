"""Coordinate-free restriction of three-dimensional STF tensors."""

import math
from functools import lru_cache

import torch
from e3nn import o3

from ..co3.basis import path_matrix as spherical_path_matrix
from ..co3.symmetric import SymmetricBasis
from .basis import path_matrix
from .projection import detracing_coefficients


@lru_cache(maxsize=None)
def restriction_scale(l, m):
    """Return the orthonormal restriction scale sqrt(binomial(2*l, l-m)/2**(l-m))."""
    if not 0 <= m <= l:
        raise ValueError("Restriction requires 0 <= m <= l.")
    return math.sqrt(math.comb(2 * l, l - m) / 2 ** (l - m))


@lru_cache(maxsize=None)
def restriction_matrix(l, m):
    """Return the normalized spherical-to-transverse map at the positive y-axis.

    Parameters
    ----------
    l, m : int
        Angular degree and order, with 0 <= m <= l.

    Returns
    -------
    torch.Tensor
        Cached CPU float64 matrix of shape (1 if m == 0 else 2, 2*l+1).
        Transverse axes are ordered (z, x). Do not modify in place.
    """
    scale = restriction_scale(l, m)
    indices = [
        sum((0 if (i >> (m - j - 1)) & 1 else 2) * 3 ** (m - j - 1) for j in range(m))
        for i in range(2**m)
    ]
    longitudinal = (3 ** (l - m) - 1) // 2
    matrix = spherical_path_matrix(l).reshape(3**m, 3 ** (l - m), 2 * l + 1)
    return scale * path_matrix(m).T @ matrix[indices, longitudinal, :]


@lru_cache(maxsize=None)
def coupling_coefficients(l1, l2, l3, normalization="component"):
    """Return reference-axis CG coefficients for every retained local order.

    Parameters
    ----------
    l1, l2, l3 : int
        An allowed input, harmonic, and output degree triple.
    normalization : {"component", "norm", "integral"}, optional
        Harmonic normalization. Instruction path weights are not included.

    Returns
    -------
    tuple of float
        Coefficients for orders zero through min(l1, l3). Odd couplings
        use the normalized transverse generator; their order-zero value is zero.
    """
    if normalization not in ("component", "norm", "integral"):
        raise ValueError("Unknown harmonic normalization.")
    cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")[:, l2, :].T
    pole = 1.0 if normalization == "norm" else math.sqrt(2 * l2 + 1)
    if normalization == "integral":
        pole /= math.sqrt(4 * math.pi)
    odd = (l1 + l2 + l3) % 2
    result = []
    for m in range(min(l1, l3) + 1):
        if odd and m == 0:
            result.append(0.0)
            continue
        block = restriction_matrix(l3, m) @ cg @ restriction_matrix(l1, m).T
        transform = (
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64, device="cpu")
            if odd
            else torch.eye(block.shape[0], dtype=torch.float64, device="cpu")
        )
        result.append(pole * float((block * transform).sum() / block.shape[0]))
    return tuple(result)


class TransverseProjector(torch.nn.Module):
    """Project a symmetric tensor onto transverse STF tensors.

    Parameters
    ----------
    m : int
        Non-negative Cartesian rank. Storage has size 3**m.
    """

    def __init__(self, m):
        super().__init__()
        if not isinstance(m, int) or m < 0:
            raise ValueError("The order must be a non-negative integer.")
        self.m = m
        self.bases = torch.nn.ModuleList(
            SymmetricBasis(rank) for rank in (range(m, -1, -2) if m > 2 else ())
        )
        self.coefficients = detracing_coefficients(m)
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, tensor, direction):
        """Project symmetric (..., 3**m) tensors along unit (..., 3) directions.

        Leading dimensions broadcast. The result retains global Cartesian
        indices; it is not stored in the two-dimensional co2.Irreps layout.
        """
        if tensor.shape[-1] != 3**self.m or direction.shape[-1] != 3:
            raise ValueError("Tensor or direction has an incorrect trailing size.")
        if self.m == 0:
            shape = torch.broadcast_shapes(tensor.shape[:-1], direction.shape[:-1])
            return tensor.expand(*shape, 1)
        plane = self.identity - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        if self.m == 1:
            return torch.einsum("...ij,...j->...i", plane, tensor)
        if self.m == 2:
            value = plane @ tensor.unflatten(-1, (3, 3)) @ plane
            trace = value.diagonal(dim1=-2, dim2=-1).sum(-1)
            return (value - 0.5 * trace[..., None, None] * plane).flatten(-2)
        basis = self.bases[0]
        traces = [basis(basis.pack(tensor), plane)]
        entries = torch.cat(
            (plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1
        )
        for basis in self.bases[:-1]:
            traces.append(basis.trace(traces[-1]))
        value = traces[-1] * self.coefficients[-1]
        for k in range(len(traces) - 2, -1, -1):
            value = self.coefficients[k] * traces[k] + self.bases[k].multiply(
                value, entries
            )
        return self.bases[0].unpack(value)

    def extra_repr(self):
        return f"m={self.m}"


class Restriction(torch.nn.Module):
    """Decompose an STF tensor into transverse irreducible tensors.

    Parameters
    ----------
    l : int
        Three-dimensional angular degree. Inputs have Cartesian size 3**l.

    Notes
    -----
    Order m uses 3**m global Cartesian entries, with one independent
    coordinate at m=0 and two at m>0. No transverse axes are selected.
    """

    def __init__(self, l):
        super().__init__()
        if not isinstance(l, int) or l < 0:
            raise ValueError("The degree must be a non-negative integer.")
        self.l = l
        self.bases = torch.nn.ModuleList(
            SymmetricBasis(m) for m in (range(l + 1) if l > 2 else ())
        )
        self.coefficients = tuple(
            tuple(abs(value) for value in detracing_coefficients(m))
            for m in range(l + 1)
        )
        self.scales = tuple(restriction_scale(l, m) for m in range(l + 1))
        self.register_buffer("identity", torch.eye(3), persistent=False)
        self.register_buffer(
            "basis",
            spherical_path_matrix(l).to(torch.get_default_dtype()).clone(),
            persistent=False,
        )

    def forward(self, tensor, direction):
        """Return orders 0,...,l for STF tensors and unit, broadcastable directions."""
        if tensor.shape[-1] != 3**self.l or direction.shape[-1] != 3:
            raise ValueError("Tensor or direction has an incorrect trailing size.")
        if self.l == 0:
            shape = torch.broadcast_shapes(tensor.shape[:-1], direction.shape[:-1])
            return (tensor.expand(*shape, 1),)
        if self.l == 1:
            scalar = (tensor * direction).sum(-1, keepdim=True)
            return scalar, tensor - scalar * direction
        plane = self.identity - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        if self.l == 2:
            tensor = tensor.unflatten(-1, (3, 3))
            vector = torch.einsum("...ij,...j->...i", tensor, direction)
            scalar = (vector * direction).sum(-1, keepdim=True)
            transverse = plane @ tensor @ plane + 0.5 * scalar.unsqueeze(-1) * plane
            return (
                self.scales[0] * scalar,
                self.scales[1] * (vector - scalar * direction),
                transverse.flatten(-2),
            )
        contractions = [self.bases[-1].pack(tensor)]
        for m in range(self.l, 0, -1):
            contractions.append(self.bases[m].contract(contractions[-1], direction))
        contractions.reverse()
        transverse = [basis(x, plane) for basis, x in zip(self.bases, contractions)]
        entries = torch.cat(
            (plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1
        )
        values = []
        for m, (basis, scale, coefficients) in enumerate(
            zip(self.bases, self.scales, self.coefficients)
        ):
            # Tr^k(U_lm) = (-1)^k U_l,m-2k for a three-dimensional STF input.
            # Nested symmetric products avoid repeated trace and permutation sums.
            k = len(coefficients) - 1
            value = coefficients[k] * transverse[m - 2 * k]
            for k in range(k - 1, -1, -1):
                value = coefficients[k] * transverse[m - 2 * k] + self.bases[
                    m - 2 * k
                ].multiply(value, entries)
            values.append(basis.unpack(value * scale))
        return tuple(values)

    def inverse(self, tensors, direction, *, project=True):
        """Reconstruct Cartesian features, optionally deferring STF projection."""
        if len(tensors) != self.l + 1:
            raise ValueError("One transverse tensor is required for every order.")
        result = None
        for m, (tensor, scale) in enumerate(zip(tensors, self.scales)):
            if tensor.shape[-1] != 3**m:
                raise ValueError("A transverse tensor has an incorrect trailing size.")
            value = scale * tensor
            result = (
                value
                if result is None
                else (direction.unsqueeze(-1) * result.unsqueeze(-2)).flatten(-2)
                + value
            )
        if self.l == 0:
            shape = torch.broadcast_shapes(result.shape[:-1], direction.shape[:-1])
            result = result.expand(*shape, 1)
        return (result @ self.basis) @ self.basis.T if project else result

    def extra_repr(self):
        return f"l={self.l}"

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.basis = spherical_path_matrix(self.l).to(self.basis).clone()
        return self


def transverse_quarter_turn(tensor, direction):
    """Apply the normalized rotation generator to a positive-order transverse STF tensor."""
    value = tensor.reshape(*tensor.shape[:-1], 3, tensor.shape[-1] // 3)
    value = torch.linalg.cross(direction.unsqueeze(-2), value.transpose(-1, -2), dim=-1)
    return value.transpose(-1, -2).flatten(-2)
