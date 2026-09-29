"""Irreducible decomposition of two-dimensional Cartesian tensors."""

import torch

from ._clebsch_gordan import clebsch_gordan_product
from .irreps import Irrep, Irreps


def path_matrices(rank):
    """Generate normalized O(2) decomposition paths.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.

    Yields
    ------
    path : tuple of Irrep
        Intermediate irreps, starting with the even scalar. Higher
        orders precede lower ones. At order zero, even precedes odd.
    matrix : torch.Tensor
        CPU float64 matrix of shape (2**rank, path[-1].dim) with
        orthonormal columns. Rows follow flattened Cartesian index order.
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    scalar = Irrep(0, 1)
    vector = Irrep(1, 0)
    stack = [((scalar,), torch.ones(1, 1, dtype=torch.float64, device="cpu"))]
    coefficients = {}
    while stack:
        path, matrix = stack.pop()
        if len(path) == rank + 1:
            yield path, matrix
            continue
        ir = path[-1]
        irrep_list = sorted(vector * ir, key=lambda ir: ir.m, reverse=True)
        for ir_out in reversed(irrep_list):
            key = (ir, ir_out)
            if key not in coefficients:
                coefficients[key] = clebsch_gordan_product(
                    torch.eye(2, dtype=torch.float64, device="cpu"),
                    vector,
                    torch.eye(ir.dim, dtype=torch.float64, device="cpu"),
                    ir,
                    ir_out,
                ).permute(1, 2, 0)
            output = torch.einsum("ac,bco->bao", matrix, coefficients[key])
            output = output.flatten(0, 1)
            output = output / output.norm(dim=0)
            stack.append((path + (ir_out,), output))


class ICTD(torch.nn.Module):
    """Decompose a two-dimensional Cartesian tensor into O(2) irreps.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to torch.get_default_dtype().
    device : torch.device or str, optional
        Buffer device. Defaults to CPU.

    Attributes
    ----------
    paths : tuple of tuple of Irrep
        Coupling paths including intermediate spatial parities.
    irreps_out : Irreps
        One multiplicity-one entry per path. Repeated irreps remain separate.
    change_of_basis : torch.Tensor
        Orthogonal matrix of shape (2**rank, 2**rank).

    Notes
    -----
    Inputs have shape (..., 2**rank) and need not be symmetric or traceless.
    Batch and channel dimensions precede the flattened Cartesian indices.
    Constants are constructed in float64 by successive CG contractions [#ictd-o2]_.
    Full matrix storage scales as 2**(2*rank).

    References
    ----------
    .. [#ictd-o2] S. Shao et al., "High-Rank Irreducible Cartesian Tensor
       Decomposition and Bases of Equivariant Spaces", JMLR 26(175), 2025.
       https://www.jmlr.org/papers/v26/25-0134.html

    Examples
    --------
    >>> decomposition = ICTD(2)
    >>> decomposition.irreps_out.dim
    4
    >>> x = torch.randn(4, 4)
    >>> torch.testing.assert_close(decomposition.inverse(decomposition(x)), x)
    """

    def __init__(self, rank, *, dtype=None, device=None):
        super().__init__()
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point dtype.")
        self.rank = rank
        self.dim = 2**rank
        change_of_basis = torch.empty(
            self.dim, self.dim, dtype=torch.float64, device="cpu"
        )
        paths, slices = [], []
        offset = 0
        for path, matrix in path_matrices(rank):
            paths.append(path)
            slices.append(slice(offset, offset + matrix.shape[1]))
            change_of_basis[:, slices[-1]] = matrix
            offset += matrix.shape[1]
        self.paths = tuple(paths)
        self.slices = tuple(slices)
        self.irreps_out = Irreps([(path[-1], 1) for path in paths])
        self.register_buffer(
            "change_of_basis",
            change_of_basis.to(dtype=dtype or torch.get_default_dtype(), device=device),
            persistent=False,
        )

    def forward(self, features):
        """Extract irreducible features from tensors of shape (..., 2**rank)."""
        return features @ self.change_of_basis

    def inverse(self, features):
        """Reconstruct flattened Cartesian tensors from irreducible features."""
        return features @ self.change_of_basis.T

    def path_matrix(self, index):
        """Return the rectangular matrix for a path index as a buffer view."""
        return self.change_of_basis[:, self.slices[index]]

    def project(self, features, index):
        """Project a flattened Cartesian tensor onto one irreducible path."""
        matrix = self.path_matrix(index)
        return (features @ matrix) @ matrix.T

    def extra_repr(self):
        return f"rank={self.rank}, {self.irreps_out}"

    def _apply(self, fn, recurse=True):
        dtype = self.change_of_basis.dtype
        super()._apply(fn, recurse)
        if torch.finfo(self.change_of_basis.dtype).eps < torch.finfo(dtype).eps:
            for index, (_, matrix) in enumerate(path_matrices(self.rank)):
                self.change_of_basis[:, self.slices[index]].copy_(
                    matrix.to(self.change_of_basis)
                )
        return self
