################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Sequence, Tuple

import torch

from .irreps import Irreps, IrrepsLike


class Layout(torch.nn.Module):
    """Convert between dense and grouped Cartesian O(3) feature layouts.

    The dense layout is ``(batch, irreps.dim, channels)``. A grouped block for
    ``mul x (l, p)`` has shape ``(batch, 3**l, mul * channels)``.
    """

    def __init__(self, irreps: IrrepsLike, channels: int) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        if not isinstance(channels, int) or isinstance(channels, bool):
            raise TypeError("channels must be an integer.")
        if channels < 1:
            raise ValueError("channels must be positive.")
        self.channels = channels

    def to_grouped(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Convert ``(batch, irreps.dim, channels)`` to grouped blocks."""
        expected = (self.irreps.dim, self.channels)
        if input.ndim != 3 or tuple(input.shape[-2:]) != expected:
            raise ValueError(
                f"input must have shape (batch, {expected[0]}, {expected[1]})."
            )
        blocks = []
        offset = 0
        for multiplicity, irrep in self.irreps:
            width = multiplicity * irrep.dim
            block = input[:, offset : offset + width, :]
            block = block.reshape(
                input.shape[0], multiplicity, irrep.dim, self.channels
            )
            blocks.append(
                block.permute(0, 2, 1, 3).reshape(
                    input.shape[0], irrep.dim, multiplicity * self.channels
                )
            )
            offset += width
        return tuple(blocks)

    def from_grouped(self, blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        """Convert grouped blocks to ``(batch, irreps.dim, channels)``."""
        if len(blocks) != len(self.irreps):
            raise ValueError("Expected one block per Irreps group.")
        dense = []
        batch = None
        for block, (multiplicity, irrep) in zip(blocks, self.irreps):
            expected = (irrep.dim, multiplicity * self.channels)
            if block.ndim != 3 or tuple(block.shape[-2:]) != expected:
                raise ValueError(
                    "Grouped block must have shape "
                    f"(batch, {expected[0]}, {expected[1]})."
                )
            if batch is None:
                batch = block.shape[0]
            elif block.shape[0] != batch:
                raise ValueError("Grouped blocks must share one batch size.")
            dense.append(
                block.reshape(batch, irrep.dim, multiplicity, self.channels)
                .permute(0, 2, 1, 3)
                .reshape(batch, multiplicity * irrep.dim, self.channels)
            )
        if dense:
            return torch.cat(dense, dim=1)
        raise ValueError("Cannot infer batch metadata from empty grouped input.")

    def extra_repr(self) -> str:
        return f"irreps={self.irreps}, channels={self.channels}"
