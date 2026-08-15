################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch

from ...linear import torchLinear
from .utils import so2_expand_index


class uvSO2MLinear(torch.nn.Module):
    """
    Based on https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/so2_ops.py
    Original Paper: https://proceedings.mlr.press/v202/passaro23a.html
    """

    def __init__(
        self,
        m: int,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: int,
        num_components_out: int,
        weight_type: str = "w1_w2",
    ):
        super().__init__()

        self.m = m
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.num_components_in = num_components_in
        self.num_components_out = num_components_out
        self.weight_type = weight_type
        assert self.num_components_in > 0
        assert self.num_components_out > 0

        if weight_type == "w1_w2":
            self.fc = torchLinear(
                self.num_components_in * self.num_channel_in,
                self.num_components_out * self.num_channel_out * 2,
                bias=False,
            )
            self.fc.weight.data.mul_(1 / math.sqrt(2))
        elif weight_type == "w1_w1":
            self.fc = torchLinear(
                self.num_components_in * self.num_channel_in,
                self.num_components_out * self.num_channel_out,
                bias=False,
            )
            self.fc.weight.data.mul_(1 / math.sqrt(2))
        else:
            self.fc = torchLinear(
                self.num_components_in * self.num_channel_in,
                self.num_components_out * self.num_channel_out,
                bias=False,
            )
        self._Cout = self.num_components_out * self.num_channel_out

    def forward(self, x, concat_outputs=True):
        # [batch, 2, -1]
        if self.weight_type == "w1_w2":
            return self.w1_w2_forward(x, concat_outputs)
        elif self.weight_type == "w1_w1":
            return self.w1_w1_forward(x, concat_outputs)
        else:
            return self.w1_forward(x, concat_outputs)

    def w1_w2_forward(
        self, x, concat_outputs=True
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        x = self.fc(x)
        w1_x = x.narrow(2, 0, self._Cout)
        w2_x = x.narrow(2, self._Cout, self._Cout)
        xr = w1_x.narrow(1, 0, 1) - w2_x.narrow(1, 1, 1)  # w1_x+m - w2x-m
        xi = w1_x.narrow(1, 1, 1) + w2_x.narrow(1, 0, 1)  # w1_x-m + w2x+m
        x_out = (xr, xi)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out

    def w1_w1_forward(self, x, concat_outputs=True):
        xr = x.narrow(1, 0, 1)
        xi = x.narrow(1, 1, 1)
        # yr = W(xr - xi)
        # yi = W(xi + xr)
        yr_in = xr - xi
        yi_in = xi + xr
        yr = self.fc(yr_in)
        yi = self.fc(yi_in)
        x_out = (yr, yi)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out

    def w1_forward(self, x, concat_outputs=True):
        x = self.fc(x)
        if concat_outputs:
            return x
        return (x.narrow(1, 0, 1), x.narrow(1, 1, 1))


class uvSO2Linear(torch.nn.Module):
    """
    Based on https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/so2_ops.py
    Original Paper: https://proceedings.mlr.press/v202/passaro23a.html
    """

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: Union[None, list[int]] = None,
        num_components_out: Union[None, list[int]] = None,
        weight_type: str = "w1_w2",  # [w1_w2, w1_w1, w1]
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.weight_type = weight_type

        if num_components_in is None:
            self.num_components_in = [lmax + 1 - m for m in range(mmax + 1)]
        else:
            self.num_components_in = num_components_in
        assert isinstance(self.num_components_in, list)
        if num_components_out is None:
            self.num_components_out = [lmax + 1 - m for m in range(mmax + 1)]
        else:
            self.num_components_out = num_components_out
        assert isinstance(self.num_components_out, list)

        self.m0_rlinear = torchLinear(
            self.num_channel_in * self.num_components_in[0],
            self.num_channel_out * self.num_components_out[0],
            bias=True,
        )
        self.ms_clinear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.ms_clinear.append(
                uvSO2MLinear(
                    m,
                    self.num_channel_in,
                    self.num_channel_out,
                    self.num_components_in[m],
                    self.num_components_out[m],
                    weight_type=weight_type,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, num_components, num_channel],
        # layout of m components is (0, 0, ...), (1, 1, ...), ...

        B = x.size(0)
        Cout = self.num_channel_out

        outputs = []

        # m = 0
        xm0 = x.narrow(1, 0, self.num_components_in[0])
        # xm0 = xm0.reshape(B, -1)
        # xm0 = self.m0_rlinear(xm0)
        # xm0 = xm0.view(B, -1, Cout)
        xm0 = xm0.reshape(B, self.num_components_in[0] * self.num_channel_in)
        xm0 = self.m0_rlinear(xm0)
        xm0 = xm0.view(B, self.num_components_out[0], Cout)
        outputs.append(xm0)

        # m > 0
        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            xm = x.narrow(1, offset, 2 * (self.num_components_in[m]))
            offset = offset + 2 * self.num_components_in[m]
            # xm = xm.reshape(B, 2, -1)
            # xm = self.ms_clinear[m - 1](xm, concat_outputs=False)
            # xr, xi = xm[0], xm[1]
            # xr = xr.view(B, -1, Cout)
            # xi = xi.view(B, -1, Cout)
            xm = xm.reshape(B, 2, self.num_components_in[m] * self.num_channel_in)
            xm = self.ms_clinear[m - 1](xm, concat_outputs=False)
            xr, xi = xm[0], xm[1]
            xr = xr.view(B, self.num_components_out[m], Cout)
            xi = xi.view(B, self.num_components_out[m], Cout)
            outputs.append(xr)
            outputs.append(xi)
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def __repr__(self) -> str:
        ins = []
        outs = []
        for m in range(self.mmax + 1):
            n1 = self.num_components_in[m]
            n2 = self.num_components_out[m]
            ins.append(f"{self.num_channel_in * n1}x{m}m")
            outs.append(f"{self.num_channel_out * n2}x{m}m")
        num_weights = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}"
            f"({'+'.join(ins)} -> "
            f"{'+'.join(outs)} | "
            f"{num_weights} weights)"
            f"(weight_type={self.weight_type})"
            f"(bias={True})"
        )


class SO2Gate(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        scalar_act,
        tensor_act,
        channel_wise: bool = False,
        gate_m0: bool = True,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.gate_m0 = gate_m0
        self.num_m0_components = lmax + 1
        gate_start = 0 if gate_m0 else 1

        if not channel_wise:
            if gate_start <= mmax:
                self.num_components, expand_index = so2_expand_index(
                    mmax,
                    lmax,
                    start=gate_start,
                )
            else:
                self.num_components = 0
                expand_index = torch.empty(0, dtype=torch.long)
        else:
            expand_index = []
            offset = 0
            for m in range(gate_start, mmax + 1):
                index = torch.arange((lmax + 1))
                index = index + offset
                expand_index.append(index)
                if m > 0:
                    expand_index.append(index)  # +- m
                offset = offset + len(index)
            if expand_index:
                expand_index = torch.cat(expand_index, dim=0).long()
            else:
                expand_index = torch.empty(0, dtype=torch.long)
            self.num_components = offset

        self.register_buffer("expand_index", expand_index, persistent=False)

        self.scalar_act = scalar_act
        self.tensor_act = tensor_act

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        g = self.tensor_act(g).view(B, self.num_components, self.num_channel)
        g = torch.index_select(g, dim=1, index=self.expand_index)
        if self.gate_m0:
            return g * x
        x_m0 = self.scalar_act(x[:, : self.num_m0_components])
        x_m = g * x[:, self.num_m0_components :]
        return torch.cat((x_m0, x_m), dim=1)

    # def __repr__(self) -> str:
    #     return (
    #         f"{self.__class__.__name__} + "
    #         f"(act='sigmoid', gate_m0={self.gate_m0})"
    #     )
