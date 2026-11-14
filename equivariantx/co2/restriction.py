"""Coordinate-free restriction of three-dimensional STF tensors."""

import math
from functools import lru_cache

import torch
from e3nn import o3

from ..co3.basis import path_matrix as spherical_path_matrix
from ..co3.symmetric import SymmetricBasis
from ..ops.precision import validate_module_precision, validate_precision
from ..ops.torch.co2 import restriction as torch_kernel
from .basis import path_matrix
from .projection import detracing_coefficients


def execution(module, tensor):
    if getattr(module, "backend", "torch") == "cuda" and tensor.is_cuda:
        from ..ops.cuda.co2 import restriction as cuda_kernel

        return cuda_kernel
    return torch_kernel


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
    backend : {"torch", "cuda"}, optional
        Execution backend. CPU tensors use Torch.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.TransverseProjector(1)
    >>> layer(torch.tensor([1., 2., 3.]), torch.tensor([0., 1., 0.])).tolist()
    [1.0, 0.0, 3.0]
    """

    def __init__(self, m, *, backend="torch"):
        super().__init__()
        if not isinstance(m, int) or m < 0:
            raise ValueError("The order must be a non-negative integer.")
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
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
        validate_module_precision(self, (tensor, direction))
        if tensor.shape[-1] != 3**self.m or direction.shape[-1] != 3:
            raise ValueError("Tensor or direction has an incorrect trailing size.")
        return execution(self, tensor).transverse(self, tensor, direction)

    def extra_repr(self):
        return f"m={self.m}"


class Restriction(torch.nn.Module):
    """Decompose an STF tensor into transverse irreducible tensors.

    Parameters
    ----------
    l : int
        Three-dimensional angular degree. Inputs have Cartesian size 3**l.
    backend : {"torch", "cuda"}, optional
        Execution backend. CPU tensors use Torch.

    Notes
    -----
    Order m uses 3**m global Cartesian entries, with one independent
    coordinate at m=0 and two at m>0. No transverse axes are selected.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.Restriction(1)
    >>> parts = layer(torch.ones(2, 3), torch.tensor([0., 1., 0.]))
    >>> tuple(part.shape[-1] for part in parts)
    (1, 3)
    """

    def __init__(self, l, *, backend="torch"):
        super().__init__()
        if not isinstance(l, int) or l < 0:
            raise ValueError("The degree must be a non-negative integer.")
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
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
        validate_module_precision(self, (tensor, direction))
        if tensor.shape[-1] != 3**self.l or direction.shape[-1] != 3:
            raise ValueError("Tensor or direction has an incorrect trailing size.")
        return execution(self, tensor).restrict(self, tensor, direction)

    def inverse(self, tensors, direction, *, project=True):
        """Reconstruct Cartesian features, optionally deferring STF projection."""
        validate_module_precision(self, (tensors, direction))
        if len(tensors) != self.l + 1:
            raise ValueError("One transverse tensor is required for every order.")
        return execution(self, tensors[0]).inverse(self, tensors, direction, project)

    def extra_repr(self):
        return f"l={self.l}"

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.basis = spherical_path_matrix(self.l).to(self.basis).clone()
        return self


def transverse_quarter_turn(tensor, direction, *, backend="torch"):
    """Apply the normalized rotation generator to a transverse STF tensor."""
    validate_precision((tensor, direction))
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    if backend == "cuda" and tensor.is_cuda:
        from ..ops.cuda.co2 import restriction as cuda_kernel

        return cuda_kernel.quarter_turn(tensor, direction)
    return torch_kernel.quarter_turn(tensor, direction)
