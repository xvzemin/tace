################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch


class GraphDropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return features
        keep_prob = 1.0 - self.drop_prob
        num_graphs = int(batch.max()) + 1 if batch.numel() else 0
        mask = keep_prob + torch.rand(
            num_graphs,
            1,
            dtype=features.dtype,
            device=features.device,
        )
        mask.floor_()
        return features * mask[batch] / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


__all__ = ["GraphDropPath"]
