################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from ..ictd import ICTD


class _SymmetricRankTwoBasisChange(torch.nn.Module):
    """Convert ``0e + 2e`` components to a Cartesian tensor."""

    def __init__(self) -> None:
        super().__init__()
        PS, DS, CS, SS = ICTD(2)
        self.register_buffer(
            "zero",
            SS[2].view(3, 3).to(dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.register_buffer(
            "two",
            SS[0].view(5, 3, 3).to(dtype=torch.get_default_dtype()),
            persistent=False,
        )
        del PS, DS, CS, SS

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum("b, ij -> bij", t0, self.zero)
        t2 = torch.einsum("bk, kij -> bij", t2, self.two)
        return t0 + t2


class DirectPolarizability(_SymmetricRankTwoBasisChange):
    """Convert direct polarizability irreps to a Cartesian tensor."""


class DirectVirials(_SymmetricRankTwoBasisChange):
    """Convert direct virial irreps to a Cartesian tensor."""


class LESQuadrupoles(_SymmetricRankTwoBasisChange):
    """Convert LES latent quadrupole irreps to a Cartesian tensor."""


class LESPolarizability(_SymmetricRankTwoBasisChange):
    """Convert LES latent polarizability irreps to a Cartesian tensor."""
