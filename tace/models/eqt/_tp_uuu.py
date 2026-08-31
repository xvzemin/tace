###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch
from e3nn import o3

from ..layout import LayoutTransform


class e3nnEqtTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        num_channel: int,
        path: list[tuple[int, int, int]],
        trainable: bool,
    ):
        super().__init__()

        self.reshap1 = LayoutTransform(irreps_in1)
        self.reshap2 = LayoutTransform(irreps_in2)
        self.reshap3 = LayoutTransform(irreps_out)

        from .equitorch.nn import TensorProduct

        self.eqt_tp = TensorProduct(
            irreps_in1="+".join(str(ir) for _, ir in irreps_in1),
            irreps_in2="+".join(str(ir) for _, ir in irreps_in2),
            irreps_out="+".join(str(ir) for _, ir in irreps_out),
            channels_in1=num_channel,
            channels_in2=num_channel,
            channels_out=num_channel,
            internal_weights=False if trainable else True,
            path_norm=True,
            trainable=trainable,
            path=[(k, i, j) for (i, j, k, _, _) in path],
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, w: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        return self.reshap3.inverse(self.eqt_tp(self.reshap1(x), self.reshap2(y), w))


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"