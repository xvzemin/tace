"""Irreducible decomposition of three-dimensional Cartesian tensors."""

import torch
from e3nn import o3
from e3nn.util import explicit_default_types

from ..ops import resolve_backend
from ..ops.common.ictd import dense_basis, initialize, path_basis, restore
from ..ops.precision import validate_module_precision
from ..ops.torch.o3 import ictd as torch_kernel


def w3j(
    l1: int, l2: int, l3: int, dtype=None, device=None, *, paths=(None, None, None)
) -> torch.Tensor:
    """Return e3nn's real Wigner 3j tensor in the selected bases.

    Parameters
    ----------
    l1, l2, l3 : int
        Angular degrees of the two inputs and the output.
    paths : tuple of three torch.Tensor or None, optional
        An orthonormal-column path matrix of shape ``(d, 2*l+1)`` maps
        that index to ``d`` Cartesian entries. ``None`` keeps it spherical.
        :func:`equivariantx.co3.path_matrix` selects the STF path;
        :func:`path_matrices` also provides paths of other tensor ranks.
    dtype : torch.dtype, optional
        Output dtype. Defaults to the PyTorch default floating-point dtype.
    device : torch.device or str, optional
        Output device. Defaults to the PyTorch default device.

    Returns
    -------
    torch.Tensor
        Coefficients of shape ``(d1, d2, d3)`` with unit Frobenius norm.
        For a single component-normalized tensor-product path, multiply
        by ``sqrt(2*l3+1)`` in either basis.

    Examples
    --------
    >>> from equivariantx import co3, o3
    >>> q = co3.path_matrix(2)
    >>> c = o3.w3j(2, 2, 2, paths=(None, q, q), dtype=q.dtype)
    >>> x, y = torch.randn(2, 5, dtype=q.dtype)
    >>> torch.einsum("i,j,ijk->k", x, q @ y, c).reshape(3, 3).shape
    torch.Size([3, 3])
    """
    if len(paths) != 3:
        raise ValueError("paths must contain one entry for each of l1, l2 and l3.")
    if all(matrix is None for matrix in paths):
        return o3.wigner_3j(l1, l2, l3, dtype=dtype, device=device)

    symbol = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
    for axis, (degree, matrix) in enumerate(zip((l1, l2, l3), paths)):
        if matrix is None:
            continue
        if not isinstance(matrix, torch.Tensor) or not matrix.is_floating_point():
            raise TypeError(
                f"paths[{axis}] must be a real floating-point tensor or None."
            )
        if matrix.ndim != 2 or matrix.shape[1] != 2 * degree + 1:
            raise ValueError(f"paths[{axis}] must have shape (d, {2 * degree + 1}).")
        symbol = (symbol.movedim(axis, -1) @ matrix.to(symbol).T).movedim(-1, axis)
    dtype, device = explicit_default_types(dtype, device)
    return symbol.to(dtype=dtype, device=device).contiguous()


def path_matrices(rank):
    """Generate normalized O(3) decomposition paths.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.

    Yields
    ------
    path : tuple of e3nn.o3.Irrep
        Intermediate irreps, starting with the even scalar. Higher
        degrees precede lower ones.
    matrix : torch.Tensor
        CPU float64 matrix of shape (3**rank, path[-1].dim) with
        orthonormal columns. Rows follow flattened Cartesian index order.

    Examples
    --------
    >>> from equivariantx import o3
    >>> [path[-1].l for path, matrix in o3.path_matrices(2)]
    [2, 1, 0]
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    scalar = o3.Irrep(0, 1)
    vector = o3.Irrep(1, -1)
    stack = [((scalar,), torch.ones(1, 1, dtype=torch.float64, device="cpu"))]
    coefficients = {}
    while stack:
        path, matrix = stack.pop()
        if len(path) == rank + 1:
            yield path, matrix
            continue
        ir = path[-1]
        irrep_list = sorted(vector * ir, key=lambda ir: ir.l, reverse=True)
        for ir_out in reversed(irrep_list):
            key = (ir, ir_out)
            if key not in coefficients:
                coefficients[key] = coupling_matrix(ir, ir_out)
            output = torch.einsum("ac,bco->bao", matrix, coefficients[key])
            output = output.flatten(0, 1)
            output = output / output.norm(dim=0)
            stack.append((path + (ir_out,), output))


def execution(module, features):
    if module.backend == "cuda" and features.is_cuda:
        from ..ops.cuda.o3 import ictd as cuda_kernel

        return cuda_kernel
    return torch_kernel


def coupling_matrix(ir, out):
    return w3j(1, ir.l, out.l, dtype=torch.float64, device="cpu")


class ICTD(torch.nn.Module):
    """Decompose a three-dimensional Cartesian tensor into O(3) irreps.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU tensors use Torch.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to the PyTorch default floating-point dtype.
    device : torch.device or str, optional
        Buffer device. Defaults to CPU.

    Attributes
    ----------
    paths : tuple of tuple of e3nn.o3.Irrep
        Coupling paths including intermediate spatial parities.
    irreps_out : e3nn.o3.Irreps
        One multiplicity-one entry per path. Repeated irreps remain separate.
    change_of_basis : torch.Tensor
        Orthogonal matrix of shape (3**rank, 3**rank), created on first access.

    Notes
    -----
    Inputs have shape (..., 3**rank) and need not be symmetric or traceless.
    Batch and channel dimensions precede the flattened Cartesian indices.
    Constants are constructed in float64 by successive CG contractions [#ictd-o3]_.
    Execution uses sparse CG stages. The optional full matrix requires 3**(2*rank) entries.
    Unlike :func:`equivariantx.co3.path_matrix`, this decomposition retains every
    irreducible path, not only the highest-degree STF subspace.

    References
    ----------
    .. [#ictd-o3] S. Shao et al., "High-Rank Irreducible Cartesian Tensor
       Decomposition and Bases of Equivariant Spaces", JMLR 26(175), 2025.
       https://www.jmlr.org/papers/v26/25-0134.html

    Examples
    --------
    >>> from equivariantx import o3
    >>> decomposition = o3.ICTD(2)
    >>> decomposition.irreps_out.dim
    9
    >>> x = torch.randn(4, 9)
    >>> torch.testing.assert_close(decomposition.inverse(decomposition(x)), x)
    """

    dimension = 3

    def __init__(self, rank, *, backend=None, dtype=None, device=None):
        backend = resolve_backend(backend, device)
        super().__init__()
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point dtype.")
        self.rank = rank
        self.dim = 3**rank
        self.backend = backend
        initialize(
            self, rank, o3.Irrep(0, 1), o3.Irrep(1, -1), coupling_matrix, dtype, device
        )
        self.irreps_out = o3.Irreps([(1, path[-1]) for path in self.paths])

    def __getattr__(self, name):
        value = super().__getattr__(name)
        if name == "change_of_basis" and value is None:
            marker = self._buffers["dtype_marker"]
            value = dense_basis(self.rank, self.dimension, marker.dtype, marker.device)
            self._buffers[name] = value
        return value

    def forward(self, features):
        """Extract irreducible features from tensors of shape (..., 3**rank)."""
        validate_module_precision(self, (features,))
        return execution(self, features).forward(self, features)

    def inverse(self, features):
        """Reconstruct flattened Cartesian tensors from irreducible features."""
        validate_module_precision(self, (features,))
        return execution(self, features).inverse(self, features)

    def path_matrix(self, index):
        """Return the orthonormal matrix for one path without building the full basis."""
        if self._buffers["change_of_basis"] is None:
            return path_basis(self, index)
        return self.change_of_basis[:, self.slices[index]]

    def project(self, features, index):
        """Project a flattened Cartesian tensor onto one irreducible path."""
        validate_module_precision(self, (features,))
        return execution(self, features).project(self, features, index)

    def extra_repr(self):
        return f"rank={self.rank}, {self.irreps_out}"

    def _apply(self, fn, recurse=True):
        dtype = self.dtype_marker.dtype
        super()._apply(fn, recurse)
        restore(self, dtype)
        return self
