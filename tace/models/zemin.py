################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Tuple

import torch
from e3nn.o3 import wigner_3j

from .ictd import ICTD


def torch_get_default_tensor_type() -> str:
    return torch.empty(0).type()


def torch_get_default_dtype() -> torch.dtype:
    """A torchscript-compatible version of torch.get_default_dtype()"""
    return torch.empty(0).dtype


def torch_get_default_device() -> torch.device:
    return torch.empty(0).device


def explicit_default_types(
    dtype: Optional[torch.dtype], device: Optional[torch.device]
) -> Tuple[torch.dtype, torch.device]:
    """A torchscript-compatible type resolver"""
    if dtype is None:
        dtype = torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device


def _cartesian_3j(l1: int, l2: int, l3: int) -> torch.Tensor:
    with torch.no_grad():
        Z = torch.einsum(
            "ijk, ai, bj, ck -> abc",
            wigner_3j(l1, l2, l3, dtype=torch.float64),
            ICTD(l1, l1, decomposition=False)[2][0],
            ICTD(l2, l2, decomposition=False)[2][0],
            ICTD(l3, l3, decomposition=False)[2][0],
        )
        Z = Z / torch.norm(Z)
    return Z


def cartesian_3j(l1: int, l2: int, l3: int, dtype=None, device=None) -> torch.Tensor:
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)

    dtype, device = explicit_default_types(dtype, device)

    Z = _cartesian_3j(l1, l2, l3)

    return Z.to(
        dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format
    )
