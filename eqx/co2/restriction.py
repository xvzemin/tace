"""Coordinate-free restriction of three-dimensional STF tensors."""

import math
from functools import lru_cache

import torch
from e3nn import o3

from ..co3.basis import path_matrix as spherical_path_matrix
from .basis import path_matrix


def tensor_power(vector, rank):
    """Return a flattened tensor power, preserving leading dimensions."""
    value = vector.new_ones((*vector.shape[:-1], 1))
    for _ in range(rank):
        value = (value.unsqueeze(-1) * vector.unsqueeze(-2)).flatten(-2)
    return value


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
        # Symmetrize by collecting equal index multisets, not by m! permutations.
        labels, indices = {}, []
        for i in range(3**m):
            digits = tuple(sorted((i // 3**j) % 3 for j in range(m)))
            indices.append(labels.setdefault(digits, len(labels)))
        indices = torch.tensor(indices, dtype=torch.long, device="cpu")
        self.num_symmetric = len(labels)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer(
            "counts",
            torch.bincount(indices),
            persistent=False,
        )
        self.coefficients = tuple(
            (-1) ** k
            * math.factorial(m)
            * math.factorial(m - k - 1)
            / (
                4**k
                * math.factorial(k)
                * math.factorial(m - 2 * k)
                * math.factorial(m - 1)
            )
            for k in range(1, m // 2 + 1)
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, tensor, direction):
        """Project symmetric (..., 3**m) tensors along unit (..., 3) directions.

        Leading dimensions broadcast. The result retains global Cartesian
        indices; it is not stored in the two-dimensional co2.Irreps layout.
        """
        if tensor.shape[-1] != 3 ** self.m or direction.shape[-1] != 3:
            raise ValueError("Tensor or direction has an incorrect trailing size.")
        if self.m == 0:
            shape = torch.broadcast_shapes(tensor.shape[:-1], direction.shape[:-1])
            return tensor.expand(*shape, 1)
        plane = self.identity - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        value = tensor
        for axis in range(self.m):
            shaped = value.reshape(
                *value.shape[:-1], 3**axis, 3, 3 ** (self.m - axis - 1)
            )
            value = torch.einsum("...aib,...ij->...ajb", shaped, plane).flatten(-3)
        output, trace = value, value
        for k, coefficient in enumerate(self.coefficients, start=1):
            rank = self.m - 2 * k
            trace = (
                trace.reshape(*trace.shape[:-1], 3**rank, 3, 3)
                .diagonal(dim1=-2, dim2=-1)
                .sum(-1)
            )
            term = (
                tensor_power(plane.flatten(-2), k).unsqueeze(-1) * trace.unsqueeze(-2)
            ).flatten(-2)
            summed = term.new_zeros((*term.shape[:-1], self.num_symmetric)).scatter_add(
                -1, self.indices.expand_as(term), term
            )
            output = output + coefficient * (summed / self.counts).index_select(
                -1, self.indices
            )
        return output

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
        self.projectors = torch.nn.ModuleList(
            TransverseProjector(m) for m in range(l + 1)
        )
        self.scales = tuple(restriction_scale(l, m) for m in range(l + 1))
        self.register_buffer(
            "basis",
            spherical_path_matrix(l).to(torch.get_default_dtype()).clone(),
            persistent=False,
        )

    def forward(self, tensor, direction):
        """Return orders 0,...,l for STF tensors and unit, broadcastable directions."""
        if tensor.shape[-1] != 3 ** self.l or direction.shape[-1] != 3:
            raise ValueError("Tensor or direction has an incorrect trailing size.")
        values = []
        for m, (projector, scale) in enumerate(zip(self.projectors, self.scales)):
            x = tensor.reshape(*tensor.shape[:-1], 3**m, 3 ** (self.l - m))
            x = (x @ tensor_power(direction, self.l - m).unsqueeze(-1)).squeeze(-1)
            values.append(projector(x, direction) * scale)
        return tuple(values)

    def inverse(self, tensors, direction, *, project=True):
        """Reconstruct Cartesian features, optionally deferring STF projection."""
        if len(tensors) != self.l + 1:
            raise ValueError("One transverse tensor is required for every order.")
        result = None
        for m, (tensor, scale) in enumerate(zip(tensors, self.scales)):
            if tensor.shape[-1] != 3**m:
                raise ValueError("A transverse tensor has an incorrect trailing size.")
            value = (
                tensor_power(direction, self.l - m).unsqueeze(-1) * tensor.unsqueeze(-2)
            ).flatten(-2)
            result = scale * value if result is None else result + scale * value
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
