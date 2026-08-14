################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Iterator, NamedTuple, Union

import torch

from ...linear import torchLinear
from .utils import satisfy, so2_expand_index


class uuLinearInstruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple[int, ...]
    path_weight: float
    m: int
    weight_mode: str
    output_slice: slice


class uuSO2Linear(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        weight_type: str = "w1",
        path_mode: str = "sum",  # TODO, concat may be have bug
        path_norm: bool = True,
        l1l2: Union[str, None] = None,
    ) -> None:
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.weight_type = weight_type
        self.path_mode = path_mode
        self.path_norm = path_norm
        if self.weight_type not in {"w1_w2", "w1_w1", "w1"}:
            raise ValueError(
                f"Unknown uuSO2Linear weight_type={weight_type!r}; "
                "expected 'w1_w2', 'w1_w1', or 'w1'."
            )
        if self.path_mode not in {"sum", "concat"}:
            raise ValueError(
                f"Unknown uuSO2Linear path_mode={path_mode!r}; "
                "expected 'sum' or 'concat'."
            )

        path_specs = []
        concat_path_specs = []
        instructions = []
        path_count = {}
        weight_offset = 0
        output_offset = 0

        def add_instruction(lin: int, lout: int, m: int, path_weight: float) -> int:
            nonlocal weight_offset, output_offset
            weight_mode = "real" if m == 0 or self.weight_type != "w1_w2" else "complex"
            path_shape = (
                (self.num_channel,) if weight_mode == "real" else (2, self.num_channel)
            )
            output_width = 1 if m == 0 else 2
            output_slice = slice(output_offset, output_offset + output_width)
            instructions.append(
                uuLinearInstruction(
                    i_in=lin,
                    i_out=lout,
                    path_shape=path_shape,
                    path_weight=path_weight,
                    m=m,
                    weight_mode=weight_mode,
                    output_slice=output_slice,
                )
            )
            offset = weight_offset
            weight_offset += math.prod(path_shape) // self.num_channel
            output_offset += output_width
            return offset

        for lout in range(self.lmax + 1):
            for lin in range(self.lmax + 1):
                if satisfy(lin, lout, l1l2):
                    for m in range(min(lout, lin, self.mmax) + 1):
                        path_count[(lout, m)] = path_count.get((lout, m), 0) + 1

        for m in range(self.mmax + 1):
            for lout in range(m, self.lmax + 1):
                for lin in range(m, self.lmax + 1):
                    if satisfy(lin, lout, l1l2):
                        if m == 0:
                            path_weight = (
                                1.0 / math.sqrt(path_count[(lout, 0)])
                                if self.path_norm
                                else 1.0
                            )
                            weight_index = add_instruction(lin, lout, m, path_weight)
                            concat_out = instructions[-1].output_slice.start
                            path_specs.append(
                                (
                                    self._component_index(lout, 0, False),
                                    self._component_index(lin, 0, False),
                                    weight_index,
                                    path_weight,
                                )
                            )
                            concat_path_specs.append(
                                (
                                    concat_out,
                                    self._component_index(lin, 0, False),
                                    weight_index,
                                    path_weight,
                                )
                            )
                        elif self.weight_type == "w1_w2":
                            path_weight = (
                                1.0 / math.sqrt(2 * path_count[(lout, m)])
                                if self.path_norm
                                else 1.0
                            )
                            weight_index = add_instruction(lin, lout, m, path_weight)
                            concat_out = instructions[-1].output_slice.start
                            w_real = weight_index
                            w_imag = weight_index + 1
                            out_r = self._component_index(lout, m, False)
                            out_i = self._component_index(lout, m, True)
                            in_r = self._component_index(lin, m, False)
                            in_i = self._component_index(lin, m, True)
                            path_specs.extend(
                                [
                                    (out_r, in_r, w_real, path_weight),
                                    (out_i, in_i, w_real, path_weight),
                                    (out_r, in_i, w_imag, -path_weight),
                                    (out_i, in_r, w_imag, path_weight),
                                ]
                            )
                            concat_path_specs.extend(
                                [
                                    (concat_out, in_r, w_real, path_weight),
                                    (concat_out + 1, in_i, w_real, path_weight),
                                    (concat_out, in_i, w_imag, -path_weight),
                                    (concat_out + 1, in_r, w_imag, path_weight),
                                ]
                            )
                        elif self.weight_type == "w1_w1":
                            path_weight = (
                                1.0 / math.sqrt(2 * path_count[(lout, m)])
                                if self.path_norm
                                else 1.0
                            )
                            weight_index = add_instruction(lin, lout, m, path_weight)
                            concat_out = instructions[-1].output_slice.start
                            out_r = self._component_index(lout, m, False)
                            out_i = self._component_index(lout, m, True)
                            in_r = self._component_index(lin, m, False)
                            in_i = self._component_index(lin, m, True)
                            path_specs.extend(
                                [
                                    (out_r, in_r, weight_index, path_weight),
                                    (out_i, in_i, weight_index, path_weight),
                                    (out_r, in_i, weight_index, -path_weight),
                                    (out_i, in_r, weight_index, path_weight),
                                ]
                            )
                            concat_path_specs.extend(
                                [
                                    (concat_out, in_r, weight_index, path_weight),
                                    (concat_out + 1, in_i, weight_index, path_weight),
                                    (concat_out, in_i, weight_index, -path_weight),
                                    (concat_out + 1, in_r, weight_index, path_weight),
                                ]
                            )
                        else:
                            path_weight = (
                                1.0 / math.sqrt(path_count[(lout, m)])
                                if self.path_norm
                                else 1.0
                            )
                            weight_index = add_instruction(lin, lout, m, path_weight)
                            concat_out = instructions[-1].output_slice.start
                            out_r = self._component_index(lout, m, False)
                            out_i = self._component_index(lout, m, True)
                            in_r = self._component_index(lin, m, False)
                            in_i = self._component_index(lin, m, True)
                            path_specs.extend(
                                [
                                    (out_r, in_r, weight_index, path_weight),
                                    (out_i, in_i, weight_index, path_weight),
                                ]
                            )
                            concat_path_specs.extend(
                                [
                                    (concat_out, in_r, weight_index, path_weight),
                                    (concat_out + 1, in_i, weight_index, path_weight),
                                ]
                            )

        self.instructions = instructions
        self.num_weights = weight_offset
        self.weight_numel = self.num_weights * self.num_channel
        self.num_components = self._num_so2_components()
        self.num_path_components = output_offset
        self.num_components_per_m = self._num_components_per_m()
        self.num_expanded_components_per_m = [
            n if m == 0 else 2 * n for m, n in enumerate(self.num_components_per_m)
        ]
        self.num_output_components = (
            self.num_components if self.path_mode == "sum" else self.num_path_components
        )
        self.path_out = torch.tensor([p[0] for p in path_specs], dtype=torch.long)
        self.path_in = torch.tensor([p[1] for p in path_specs], dtype=torch.long)
        self.path_weight_index = torch.tensor(
            [p[2] for p in path_specs], dtype=torch.long
        )
        self.path_weight = torch.tensor(
            [p[3] for p in path_specs],
            dtype=torch.get_default_dtype(),
        )
        self.concat_path_out = torch.tensor(
            [p[0] for p in concat_path_specs],
            dtype=torch.long,
        )
        self.concat_path_in = torch.tensor(
            [p[1] for p in concat_path_specs],
            dtype=torch.long,
        )
        self.concat_path_weight_index = torch.tensor(
            [p[2] for p in concat_path_specs],
            dtype=torch.long,
        )
        self.concat_path_weight = torch.tensor(
            [p[3] for p in concat_path_specs],
            dtype=torch.get_default_dtype(),
        )

    def _num_so2_components(self) -> int:
        return (self.lmax + 1) + sum(
            2 * (self.lmax + 1 - m) for m in range(1, self.mmax + 1)
        )

    def _num_components_per_m(self) -> list[int]:
        if self.path_mode == "sum":
            return [self.lmax + 1 - m for m in range(self.mmax + 1)]
        counts = [0 for _ in range(self.mmax + 1)]
        for ins in self.instructions:
            counts[ins.m] += 1
        return counts

    def _component_index(self, l: int, m: int, imag: bool) -> int:
        if m == 0:
            return l
        offset = self.lmax + 1
        for mm in range(1, m):
            offset += 2 * (self.lmax + 1 - mm)
        n = self.lmax + 1 - m
        index = offset + (l - m)
        if imag:
            index += n
        return index

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.num_channel:
            raise ValueError(
                f"uuSO2Linear expected {self.num_channel} channels, got {x.size(-1)}"
            )
        batch = x.size(0)
        shared_weight = weight.dim() == 1
        if not shared_weight and weight.size(0) == 1 and batch != 1:
            weight = weight.expand(batch, -1)
        if not shared_weight and weight.size(0) != batch:
            raise ValueError(
                "uuSO2Linear weight batch dimension must be 1 or match input"
            )
        if weight.numel() == self.weight_numel:
            weight = weight.view(self.num_weights, self.num_channel)
            shared_weight = True
        elif weight.size(-1) == self.weight_numel:
            weight = weight.view(batch, self.num_weights, self.num_channel)
            shared_weight = False
        else:
            raise ValueError(
                f"uuSO2Linear expected {self.weight_numel} weights, "
                f"got {weight.size(-1)}"
            )

        if self.path_mode == "sum":
            out_width = self.num_components
            path_out = self.path_out.to(device=x.device)
            path_in = self.path_in.to(device=x.device)
            path_weight_index = self.path_weight_index.to(device=x.device)
            path_weight = self.path_weight.to(dtype=x.dtype, device=x.device)
        else:
            out_width = self.num_path_components
            path_out = self.concat_path_out.to(device=x.device)
            path_in = self.concat_path_in.to(device=x.device)
            path_weight_index = self.concat_path_weight_index.to(device=x.device)
            path_weight = self.concat_path_weight.to(dtype=x.dtype, device=x.device)

        x_path = torch.index_select(x, dim=1, index=path_in)
        if shared_weight:
            w_path = torch.index_select(weight, dim=0, index=path_weight_index)
            y_path = x_path * w_path.unsqueeze(0)
        else:
            w_path = torch.index_select(weight, dim=1, index=path_weight_index)
            y_path = x_path * w_path
        y_path = y_path * path_weight.view(1, -1, 1)
        out = x.new_zeros(batch, out_width, self.num_channel)
        out.index_add_(1, path_out, y_path)
        return out

    def _get_weights(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if weight.size(-1) != self.weight_numel:
            raise ValueError(
                f"uuSO2Linear expected {self.weight_numel} weights, "
                f"got {weight.size(-1)}"
            )
        return weight

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        offset = sum(
            math.prod(ins.path_shape) for ins in self.instructions[:instruction]
        )
        ins = self.instructions[instruction]
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        flat_size = math.prod(ins.path_shape)
        return weight.narrow(-1, offset, flat_size).view(batchshape + ins.path_shape)

    def weight_views(
        self,
        weight: torch.Tensor,
        yield_instruction: bool = False,
    ) -> Iterator[torch.Tensor]:
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            flat_size = math.prod(ins.path_shape)
            this_weight = weight.narrow(-1, offset, flat_size).view(
                batchshape + ins.path_shape
            )
            offset += flat_size
            if yield_instruction:
                yield ins_i, ins, this_weight
            else:
                yield this_weight

    def extra_repr(self) -> str:
        return (
            f"mmax={self.mmax}, lmax={self.lmax}, "
            f"num_channel={self.num_channel}, mode=uu, path_norm={self.path_norm}, "
            f"weight_type={self.weight_type}, "
            # f"num_weights={self.num_weights}, "
            f"num_paths={self.path_out.numel()}, "
            # f"num_components_per_m={self.num_components_per_m}, "
            # f"num_output_components={self.num_output_components}}"
            f"weight_numel={self.weight_numel}"
        )


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
