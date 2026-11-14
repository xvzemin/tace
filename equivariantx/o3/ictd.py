"""Irreducible decomposition of three-dimensional Cartesian tensors."""

import torch
from e3nn import o3

from ..ops import resolve_backend
from ..ops.common.ictd import dense_basis, initialize, restore
from ..ops.precision import validate_module_precision
from ..ops.torch.o3 import ictd as torch_kernel


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
                coefficients[key] = o3.wigner_3j(
                    1, ir.l, ir_out.l, dtype=torch.float64, device="cpu"
                )
            output = torch.einsum("ac,bco->bao", matrix, coefficients[key])
            output = output.flatten(0, 1)
            output = output / output.norm(dim=0)
            stack.append((path + (ir_out,), output))


def execution(module, features):
    if getattr(module, "backend", "torch") == "cuda" and features.is_cuda:
        from ..ops.cuda.o3 import ictd as cuda_kernel

        return cuda_kernel
    return torch_kernel


def coupling_matrix(ir, out):
    return o3.wigner_3j(1, ir.l, out.l, dtype=torch.float64, device="cpu")


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
        Buffer dtype.get_default_dtype().
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
        """Return the rectangular matrix for a path index as a buffer view."""
        return self.change_of_basis[:, self.slices[index]]

    def project(self, features, index):
        """Project a flattened Cartesian tensor onto one irreducible path."""
        validate_module_precision(self, (features,))
        return execution(self, features).project(self, features, index)

    def extra_repr(self):
        return f"rank={self.rank}, {self.irreps_out}"

    def _apply(self, fn, recurse=True):
        marker = self._buffers.get("dtype_marker")
        dtype = self.change_of_basis.dtype if marker is None else marker.dtype
        super()._apply(fn, recurse)
        restore(self, dtype)
        return self
