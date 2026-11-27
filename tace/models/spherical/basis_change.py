################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from equivariantx.o3 import path_matrices


class _SymmetricRankTwoBasisChange(torch.nn.Module):
    """Convert ``0e + 2e`` components to a Cartesian tensor."""

    float64_buffers = ("zero", "two")

    def __init__(self) -> None:
        super().__init__()
        basis = {path[-1].l: q for path, q in path_matrices(2)}
        self.register_buffer("zero", basis[0].reshape(3, 3), persistent=False)
        self.register_buffer("two", basis[2].T.reshape(5, 3, 3), persistent=False)

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum("b, ij -> bij", t0, self.zero)
        t2 = torch.einsum("bk, kij -> bij", t2, self.two)
        return t0 + t2


class DirectPolarizability(_SymmetricRankTwoBasisChange):
    """Convert direct polarizability irreps to a Cartesian tensor."""


class DirectVirials(_SymmetricRankTwoBasisChange):
    """Convert direct virial irreps to a Cartesian tensor."""


class LESLatentQuads(_SymmetricRankTwoBasisChange):
    """Convert LES ``latent_quads`` irreps to a Cartesian tensor."""


class LESLatentAlphas(_SymmetricRankTwoBasisChange):
    """Convert LES ``latent_alphas`` irreps to a Cartesian tensor."""
