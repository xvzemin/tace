################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from functools import lru_cache
from typing import Optional

import torch
from e3nn import o3


@lru_cache(maxsize=None)
def _path_matrix(degree: int) -> torch.Tensor:
    """Build the orthonormal Cartesian path matrix for one degree."""
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a non-negative integer.")
    basis = o3.wigner_3j(0, 0, 0, dtype=torch.float64)
    for order in range(degree):
        coefficient = o3.wigner_3j(
            1,
            order,
            order + 1,
            dtype=torch.float64,
        )
        basis = torch.einsum("abc,dce->dabe", basis, coefficient)
        basis = basis.reshape(coefficient.shape[0], -1, coefficient.shape[-1])
    basis = basis.reshape(-1, basis.shape[-1])
    return basis * basis.square().sum(0)[0].rsqrt()


@lru_cache(maxsize=None)
def _path_matrix_on(
    degree: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _path_matrix(degree).to(dtype=dtype, device=device)


def path_matrix(
    degree: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the Cartesian-to-irreducible path matrix.

    Parameters
    ----------
    degree : int
        Non-negative Cartesian tensor rank.
    dtype : torch.dtype, optional
        Requested matrix dtype.
    device : torch.device, optional
        Requested matrix device.

    Returns
    -------
    torch.Tensor
        Orthonormal matrix with shape ``(3**degree, 2 * degree + 1)``.
    """
    dtype = torch.float64 if dtype is None else dtype
    device = torch.device("cpu") if device is None else torch.device(device)
    return _path_matrix_on(degree, dtype, device)


def project(input: torch.Tensor, degree: int) -> torch.Tensor:
    """Project ambient Cartesian tensors onto their symmetric traceless part.

    Parameters
    ----------
    input : torch.Tensor
        Tensor with shape ``(..., 3**degree, channels)``. The penultimate axis
        stores flattened ambient Cartesian components.
    degree : int
        Non-negative Cartesian tensor rank.

    Returns
    -------
    torch.Tensor
        Projected tensor with the same shape, lying in the degree-``degree``
        symmetric traceless subspace.
    """
    if input.ndim < 2 or input.shape[-2] != 3**degree:
        raise ValueError(f"input must have Cartesian dimension {3**degree} at axis -2.")
    matrix = path_matrix(degree, dtype=input.dtype, device=input.device)
    irreducible = torch.einsum("md,...dc->...mc", matrix.T, input)
    return torch.einsum("dm,...mc->...dc", matrix, irreducible)


def project_irreps(input: torch.Tensor, irreps) -> torch.Tensor:
    """Project flattened ``ir_mul`` entries onto irreducible subspaces.

    Parameters
    ----------
    input : torch.Tensor
        Ambient Cartesian features with shape ``(..., irreps.dim)``.
    irreps : IrrepsLike
        Representation metadata for the flattened trailing dimension.

    Returns
    -------
    torch.Tensor
        Projected features with the same shape and layout.
    """
    from .irreps import Irreps

    irreps = Irreps(irreps)
    if input.ndim < 1 or input.size(-1) != irreps.dim:
        raise ValueError(
            f"input trailing dimension must be {irreps.dim}, got {tuple(input.shape)}."
        )
    outputs = []
    for (ir, mul), ir_slice in zip(irreps, irreps.slices()):
        values = input[..., ir_slice].reshape(*input.shape[:-1], ir.dim, mul)
        outputs.append(
            project(values, ir.l).reshape(*input.shape[:-1], ir.dim * mul)
        )
    if outputs:
        return torch.cat(outputs, dim=-1)
    return input.new_empty(*input.shape[:-1], 0)
