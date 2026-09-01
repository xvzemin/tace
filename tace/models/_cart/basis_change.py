################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch


class _SymmetricRankTwoBasisChange(torch.nn.Module):
    """Combine Cartesian ``0e`` and ambient ``2e`` components."""

    def forward(self, scalar: torch.Tensor, traceless: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(3, dtype=scalar.dtype, device=scalar.device)
        scalar_tensor = scalar[..., None, None] * identity / math.sqrt(3.0)
        return scalar_tensor + traceless.reshape(*traceless.shape[:-1], 3, 3)


class DirectPolarizability(_SymmetricRankTwoBasisChange):
    pass


class DirectVirials(_SymmetricRankTwoBasisChange):
    pass


class LESLatentQuads(_SymmetricRankTwoBasisChange):
    pass


class LESLatentAlphas(_SymmetricRankTwoBasisChange):
    pass


__all__ = [
    "DirectPolarizability",
    "DirectVirials",
    "LESLatentAlphas",
    "LESLatentQuads",
]
