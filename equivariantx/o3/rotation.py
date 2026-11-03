"""Infinitesimal rotations in the real spherical basis."""

import math
from functools import lru_cache

import torch
from e3nn import o3


@lru_cache(maxsize=64)
def so3_generators(l):
    """Return the rotation generators for angular degree ``l``.

    Parameters
    ----------
    l : int
        Non-negative angular degree.

    Returns
    -------
    torch.Tensor
        Cached CPU float64 tensor of shape ``(3, 2 * l + 1, 2 * l + 1)``.
        Matrices generate rotations about x, y, and z in the real spherical
        basis. The returned tensor must not be modified in place.
    """
    if not isinstance(l, int) or l < 0:
        raise ValueError("l must be a non-negative integer.")
    if l == 0:
        return torch.zeros(3, 1, 1, dtype=torch.float64, device="cpu")
    value = -math.sqrt(l * (l + 1) * (2 * l + 1)) * o3.wigner_3j(
        l, 1, l, dtype=torch.float64, device="cpu"
    ).permute(1, 2, 0)
    return value * value[1, l - 1, l + 1].sign()
