"""Irreducible decomposition of Cartesian tensor powers."""

import torch
from e3nn import o3

from .o2._clebsch_gordan import clebsch_gordan_product
from .o2.irreps import Irrep, Irreps


def path_matrices(rank, d=3):
    """Generate all normalized Cartesian decomposition paths.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.
    d : {2, 3}, optional
        Cartesian dimension.

    Yields
    ------
    path : tuple of Irrep
        Intermediate irreps, starting with the even scalar. Higher orders
        precede lower orders; at order zero in O(2), even precedes odd.
    matrix : torch.Tensor
        CPU float64 matrix of shape ``(d**rank, path[-1].dim)`` with
        orthonormal columns. Rows follow flattened Cartesian index order.
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    if not isinstance(d, int) or d not in (2, 3):
        raise ValueError("d must be 2 or 3.")
    scalar = o3.Irrep(0, 1) if d == 3 else Irrep(0, 1)
    vector = o3.Irrep(1, -1) if d == 3 else Irrep(1, 0)
    stack = [((scalar,), torch.ones(1, 1, dtype=torch.float64, device="cpu"))]
    coefficients = {}
    while stack:
        path, matrix = stack.pop()
        if len(path) == rank + 1:
            yield path, matrix
            continue
        ir = path[-1]
        irrep_list = sorted(
            vector * ir, key=lambda ir: ir.l if d == 3 else ir.m, reverse=True
        )
        for ir_out in reversed(irrep_list):
            key = (ir, ir_out)
            if key not in coefficients:
                if d == 3:
                    cg = o3.wigner_3j(
                        1, ir.l, ir_out.l, dtype=torch.float64, device="cpu"
                    )
                else:
                    # Basis-vector products give the coupling tensor in the
                    # same real convention as the native O(2) tensor product.
                    cg = clebsch_gordan_product(
                        torch.eye(2, dtype=torch.float64, device="cpu"),
                        vector,
                        torch.eye(ir.dim, dtype=torch.float64, device="cpu"),
                        ir,
                        ir_out,
                    ).permute(1, 2, 0)
                coefficients[key] = cg
            output = torch.einsum("ac,bco->bao", matrix, coefficients[key])
            output = output.flatten(0, 1)
            output = output / output.norm(dim=0)
            stack.append((path + (ir_out,), output))


class ICTD(torch.nn.Module):
    """Decompose a Cartesian tensor into all irreducible paths.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.
    d : {2, 3}, optional
        Cartesian dimension. Defaults to 3.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to ``torch.get_default_dtype()``.
    device : torch.device or str, optional
        Buffer device. Defaults to CPU.

    Attributes
    ----------
    paths : tuple of tuple of Irrep
        Coupling paths including their intermediate spatial parities.
    irreps_out : Irreps
        One entry per path, in the same order as ``paths``. Repeated irreps
        remain separate. Uses O(2) irreps for d=2 and O(3) irreps for d=3.
    change_of_basis : torch.Tensor
        Orthogonal matrix of shape ``(d**rank, d**rank)`` whose columns
        concatenate the normalized path matrices.

    Notes
    -----
    Inputs are arbitrary Cartesian tensors, not necessarily symmetric or
    traceless. Flatten the Cartesian indices into the last dimension; any
    batch and channel dimensions precede it. All paths have multiplicity
    one, so their compact feature layouts coincide.

    The matrix is constructed in float64 by successive CG contractions [1]_.
    Storage scales as ``d**(2*rank)``. Individual Cartesian projectors are
    applied through rectangular path matrices rather than stored explicitly.

    References
    ----------
    .. [1] S. Shao et al., "High-Rank Irreducible Cartesian Tensor
       Decomposition and Bases of Equivariant Spaces", JMLR 26(175), 2025,
       Algorithm 1. https://www.jmlr.org/papers/v26/25-0134.html

    Examples
    --------
    >>> decomposition = ICTD(2, d=3)
    >>> decomposition.irreps_out
    1x2e+1x1e+1x0e
    >>> x = torch.randn(4, 9)
    >>> h = decomposition(x)
    >>> torch.testing.assert_close(decomposition.inverse(h), x)
    """

    def __init__(self, rank, d=3, *, dtype=None, device=None):
        super().__init__()
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        if not isinstance(d, int) or d not in (2, 3):
            raise ValueError("d must be 2 or 3.")
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point dtype.")
        self.rank = rank
        self.d = d
        self.dim = d**rank
        change_of_basis = torch.empty(
            self.dim, self.dim, dtype=torch.float64, device="cpu"
        )
        paths, slices = [], []
        offset = 0
        for path, matrix in path_matrices(rank, d):
            paths.append(path)
            slices.append(slice(offset, offset + matrix.shape[1]))
            change_of_basis[:, slices[-1]] = matrix
            offset += matrix.shape[1]
        self.paths = tuple(paths)
        self.slices = tuple(slices)
        self.irreps_out = (
            o3.Irreps([(1, path[-1]) for path in paths])
            if d == 3
            else Irreps([(path[-1], 1) for path in paths])
        )
        self.register_buffer(
            "change_of_basis",
            change_of_basis.to(dtype=dtype or torch.get_default_dtype(), device=device),
            persistent=False,
        )

    def forward(self, features):
        """Extract irreducible features from tensors of shape ``(..., d**rank)``."""
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
        return f"rank={self.rank}, d={self.d}, {self.irreps_out}"

    def _apply(self, fn, recurse=True):
        dtype = self.change_of_basis.dtype
        super()._apply(fn, recurse)
        if torch.finfo(self.change_of_basis.dtype).eps < torch.finfo(dtype).eps:
            # Regenerate constants instead of promoting rounded buffer values.
            for index, (_, matrix) in enumerate(path_matrices(self.rank, self.d)):
                self.change_of_basis[:, self.slices[index]].copy_(
                    matrix.to(self.change_of_basis)
                )
        return self
