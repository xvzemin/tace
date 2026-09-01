################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from functools import lru_cache

import torch
from e3nn import o3


@lru_cache(maxsize=None)
def _cartesian_to_spherical_basis(degree: int) -> torch.Tensor:
    """Build an orthonormal highest-weight Cartesian basis."""
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
        raise ValueError(
            f"input must have Cartesian dimension {3**degree} at axis -2."
        )
    basis = _cartesian_to_spherical_basis(degree).to(input)
    spherical = torch.einsum("...dc,dm->...mc", input, basis)
    return torch.einsum("...mc,dm->...dc", spherical, basis)
