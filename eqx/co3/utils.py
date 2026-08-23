################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch


def delta(
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the rank-two Kronecker delta in three dimensions."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.eye(3, dtype=dtype, device=device)


def levi_civita(
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the rank-three Levi-Civita tensor in three dimensions."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    epsilon = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    return epsilon
