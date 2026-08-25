################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch

from tace.utils.torch_scatter import scatter_sum
from ..layout import LayoutTransform
from ..linear import torchLinear
from ..softmax import GraphSoftmax
from .paths import satisfy


def so2_expand_index(mmax: int, lmax: int, start: int = 0) -> tuple[int, torch.Tensor]:
    expand_index = []
    offset = 0
    for m in range(start, mmax + 1):
        index = torch.arange((lmax + 1 - m))
        index = index + offset
        expand_index.append(index)
        if m > 0:
            expand_index.append(index)  # +- m
        offset = offset + len(index)
    expand_index = torch.cat(expand_index, dim=0)
    expand_index = expand_index.long()
    num_m_components = offset
    return num_m_components, expand_index


def so3_expand_index(mmax: int, lmax: int) -> tuple[int, torch.Tensor]:
    assert mmax == lmax
    expand_index = torch.zeros([((lmax + 1) ** 2)]).long()
    start_idx = 0
    for l in range(lmax + 1):
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
        start_idx = start_idx + length
    num_l_components = lmax + 1
    return num_l_components, expand_index


class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax (int):             Maximum degree of the spherical harmonics
        mmax (int):             Maximum order of the spherical harmonics
        use_rotate_inv_rescale (bool):
                                Whether to pre-compute inverse rotation rescale matrices
    """

    def __init__(self, lmax, mmax, use_rotate_inv_rescale=False):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.use_rotate_inv_rescale = use_rotate_inv_rescale

        m_complex = []  # this m belongs to which SO(3) m
        l_harmonic = []  # this m belongs to which SO(3) l
        m_harmonic = []  # this m belongs to which SO(2) m

        for l in range(0, self.lmax + 1):
            mmax = min(self.mmax, l)
            m = torch.arange(-mmax, mmax + 1).long()
            m_complex.append(m)
            m_harmonic.append(torch.abs(m).long())
            l_harmonic.append(torch.fill(m, l))
        m_complex = torch.cat(
            m_complex, dim=0
        )  # tensor([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3])
        m_harmonic = torch.cat(
            m_harmonic, dim=0
        )  # tensor([0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3])
        l_harmonic = torch.cat(
            l_harmonic, dim=0
        )  # tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])

        num_components = len(l_harmonic)
        to_m = torch.zeros([num_components, num_components])

        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("to_m", to_m)

        # for `torch.compile()` compatibility
        self.pre_compute_coefficient_idx()
        if self.use_rotate_inv_rescale:
            self.pre_compute_rotate_inv_rescale()

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = self.lmax

        indices = torch.arange(len(l_harmonic))
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    def pre_compute_coefficient_idx(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask = torch.bitwise_and(self.l_harmonic.le(l), self.m_harmonic.le(m))
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer(
                    "coefficient_idx_l{}_m{}".format(l, m), mask_indices
                )
        return

    def prepare_coefficient_idx(self) -> list[list[torch.Tensor]]:
        # idx = lmax, mmax
        coefficient_idx_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(
                    getattr(self, "coefficient_idx_l{}_m{}".format(l, m), None)
                )
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    def coefficient_idx(self, lmax, mmax):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
            indices = torch.arange(len(mask), device=mask.device)
            mask_indices = torch.masked_select(indices, mask)
            return mask_indices
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]

    def pre_compute_rotate_inv_rescale(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones(
                    (1, int((l + 1) ** 2), int((l + 1) ** 2))
                )
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub**2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[
                        :,
                        start_idx : (start_idx + length),
                        start_idx : (start_idx + length),
                    ] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer(
                    "rotate_inv_rescale_l{}_m{}".format(l, m), rotate_inv_rescale
                )
        return

    def prepare_rotate_inv_rescale(self):
        rotate_inv_rescale_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(
                    getattr(self, "rotate_inv_rescale_l{}_m{}".format(l, m), None)
                )
            rotate_inv_rescale_list.append(l_list)
        return rotate_inv_rescale_list

    def get_rotate_inv_rescale(self, lmax, mmax):
        temp = self.prepare_rotate_inv_rescale()
        return temp[lmax][mmax]

    def __repr__(self):
        return f"{self.__class__.__name__}(mmax={self.mmax}, lmax={self.lmax})"


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


class uuuSO2TensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        m1m2: Union[str, None] = None,
        internal_weights: bool = True,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channels = num_channels
        self.m1m2 = m1m2
        self.instructions = []

        self.num_paths = 0
        weight_numel = 0
        for m3 in range(mmax + 1):
            paths = self.enumerate_paths(m3)
            self.instructions.append(paths)
            weight_numel += num_channels * (lmax + 1) * len(paths)
            self.num_paths += len(paths)
        self.weight_numel = weight_numel

        if internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(1, self.weight_numel))
        else:
            self.register_buffer("weight", None)
        self.internal_weights = internal_weights

        output_scales = []
        n = lmax + 1
        # m = 0
        scale0 = torch.full((n,), 1.0 / math.sqrt(len(self.instructions[0])))
        output_scales.append(scale0)
        # m > 0
        for m3 in range(1, mmax + 1):
            scale = 1.0 / math.sqrt(len(self.instructions[m3]))
            output_scales.append(torch.full((2 * n,), scale))
        output_scales = torch.cat(output_scales)
        self.register_buffer("output_scales", output_scales, persistent=False)

    def enumerate_paths(self, m3: int) -> list[tuple[int, int, str]]:
        paths = []

        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                if satisfy(m1, m2, self.m1m2):
                    # x1 * x2
                    if m1 + m2 == m3:
                        paths.append((m1, m2, "sum"))
                    # x1 * conj(x2)
                    elif abs(m1 - m2) == m3:
                        paths.append((m1, m2, "diff"))

        return paths

    def rmul(self, x, y):
        # [B, n, C] * [B, n, C] =>  [B, n, C]
        z = x * y
        return z

    def cmul(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        # [B, 2, n, C] * [B, 2, n, C] => [B, 2, n, C]
        a = x[:, 0]
        b = x[:, 1]
        c = y[:, 0]
        d = y[:, 1]

        if mode == "sum":
            real = a * c - b * d
            imag = a * d + b * c
        else:
            real = a * c + b * d
            imag = b * c - a * d

        B = real.size(0)
        C = real.size(-1)
        n = self.lmax + 1

        real = real.reshape(B, n, C)
        imag = imag.reshape(B, n, C)

        out = torch.stack([real, imag], dim=1)

        return out

    def to_list(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = []
        offset = 0
        n = self.lmax + 1
        # m = 0
        out.append(x[:, offset : offset + n])
        offset += n
        # m > 0
        for m in range(1, self.mmax + 1):
            xm = x[:, offset : offset + 2 * n]
            xm = xm.view(B, 2, n, self.num_channels)
            out.append(xm)
            offset += 2 * n
        return out

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        xs = self.to_list(x)  #  m = 0 [B, lmax+1, C]
        ys = self.to_list(y)  #  m > 0 [B, 2, lmax+1, C]
        if self.internal_weights:
            ws = self.weight
        else:
            ws = weight

        C = self.num_channels

        outputs = []
        w_offset = 0

        # m = 0
        n = self.lmax + 1
        m0 = 0.0
        w_numel = C * n

        for m1, m2, mode in self.instructions[0]:
            w = ws[:, w_offset : w_offset + w_numel]  # [B, C] or [1, C]
            w = w.view(-1, n, C)
            w_offset += w_numel

            # 0 x 0
            if m1 == 0 and m2 == 0:
                z = self.rmul(xs[0], ys[0])
                out = z * w
                m0 = m0 + out

            # m > 0 and m1 -m2 = 0
            elif m1 > 0 and m2 > 0:
                z = self.cmul(xs[m1], ys[m2], "diff")
                out = z[:, 0] * w  # imag is also invariant, but nod add here
                m0 = m0 + out

        outputs.append(m0)

        # m > 0
        for m3 in range(1, self.mmax + 1):
            real = 0.0
            imag = 0.0
            for m1, m2, mode in self.instructions[m3]:
                w = ws[:, w_offset : w_offset + w_numel]
                w_offset += w_numel
                w = w.view(-1, 1, n, C)

                if m1 == 0:
                    z = xs[m1].unsqueeze(1) * ys[m2]
                elif m2 == 0:
                    z = xs[m1] * ys[m2].unsqueeze(1)
                else:
                    if m1 < m2 and mode == "diff":
                        z = self.cmul(ys[m2], xs[m1], mode)
                    else:
                        z = self.cmul(xs[m1], ys[m2], mode)

                out = z * w
                real = real + out[:, 0]
                imag = imag + out[:, 1]

            outputs.append(real)
            outputs.append(imag)

        out = torch.cat(outputs, dim=1)
        out = out * self.output_scales.view(1, -1, 1)
        return out

    def __repr__(self):
        lines = []
        lines.append(f"{self.__class__.__name__}(")
        # lines.append(
        #     f"  mmax={self.mmax}, "
        #     f"lmax={self.lmax}, "
        #     f"channels={self.num_channels}, "
        #     f"weights={self.weight_numel}"
        # )
        # lines.append("")
        lines.append("  instructions:")
        total_paths = 0
        for m3, paths in enumerate(self.instructions):
            total_paths += len(paths)
            path_strs = []
            for m1, m2, mode in paths:
                if mode == "sum":
                    expr = f"{m1}+{m2}"
                else:
                    expr = f"{m1}-{m2}"
                path_strs.append(expr)
            joined = ", ".join(path_strs)
            lines.append(f"    m={m3:<2} : {len(paths):<2} paths | {joined}")
        # lines.append("")
        lines.append(f"  total_paths={total_paths}")
        lines.append(")")
        return "\n".join(lines)


class ComplexProductBasis(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        m1m2: Union[str, None] = ">=",
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.m1m2 = m1m2
        self.tp = uuuSO2TensorProduct(
            self.mmax,
            self.lmax,
            self.num_channel,
            m1m2=self.m1m2,
            internal_weights=False,
        )
        self.weight_numel = self.tp.weight_numel

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, ws: torch.Tensor
    ) -> torch.Tensor:
        return self.tp(x, y, ws)


class uvSO2Convolution(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        num_head: int,
        use_temperature: bool,
        edge_ace_hidden: int,
        edge_wise_hidden: int,
        num_radial_basis: int,
        so2_linear_type: str,
        gate_m0: bool,
        use_asymmetric_contraction: bool,
        use_radial_rotary_attention: bool,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
        scalar_act: torch.nn.Module,
        tensor_act: torch.nn.Module,
        use_radial_phase: bool,
    ) -> None:
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_head = num_head
        self.edge_ace_hidden = edge_ace_hidden
        self.edge_wise_hidden = edge_wise_hidden or self.num_channel
        self.num_channel_per_head = self.edge_wise_hidden // self.num_head
        assert self.edge_wise_hidden % self.num_head == 0
        self.so2_linear_type = so2_linear_type
        self.use_temperature = use_temperature
        self.use_radial_rotary_attention = use_radial_rotary_attention
        self.use_asymmetric_contraction = use_asymmetric_contraction
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        self.use_radial_phase = use_radial_phase

        self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
        self.weight_numel = self.num_components * self.num_channel * 2
        self.register_buffer("expand_index", expand_index, persistent=False)

        start_m = 0 if gate_m0 else 1
        if self.use_asymmetric_contraction:
            self.num_gates = sum(lmax + 1 for _ in range(start_m, mmax + 1))
            num_components_out = [self.num_gates + lmax + 1] + [
                lmax + 1 for m in range(1, mmax + 1)
            ]
            num_components_in = [lmax + 1] + [lmax + 1 for m in range(1, mmax + 1)]
            self.split_list = [
                self.num_gates,
                (lmax + 1) + sum((lmax + 1) * 2 for m in range(1, mmax + 1)),
            ]
        else:
            self.num_gates = sum(lmax + 1 - m for m in range(start_m, mmax + 1))
            num_components_out = [self.num_gates + lmax + 1] + [
                lmax + 1 - m for m in range(1, mmax + 1)
            ]
            num_components_in = [lmax + 1] + [lmax + 1 - m for m in range(1, mmax + 1)]
            self.split_list = [
                self.num_gates,
                (lmax + 1) + sum((lmax + 1 - m) * 2 for m in range(1, mmax + 1)),
            ]

        self.linear_up = uvSO2Linear(
            mmax,
            lmax,
            self.num_channel * 2,
            self.edge_ace_hidden
            if self.use_asymmetric_contraction
            else self.edge_wise_hidden,
            num_components_out=num_components_out,
            weight_type=self.so2_linear_type,
        )
        self.nonlinearity = SO2Gate(
            mmax,
            lmax,
            self.edge_ace_hidden
            if self.use_asymmetric_contraction
            else self.edge_wise_hidden,
            channel_wise=self.use_asymmetric_contraction,
            gate_m0=gate_m0,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
        )
        if self.use_asymmetric_contraction:
            self.linear_glu = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel * 2,
                self.edge_ace_hidden
                if self.use_asymmetric_contraction
                else self.edge_wise_hidden,
                num_components_out=[lmax + 1] + [lmax + 1 for m in range(1, mmax + 1)],
                weight_type=self.so2_linear_type,
            )
            self.ece = ComplexProductBasis(
                mmax,
                lmax,
                self.edge_ace_hidden,
                m1m2=">=",
            )
            self.linear_coefs = uvSO2Linear(
                0,
                lmax,
                self.num_channel * 2,
                1,
                num_components_out=[self.ece.weight_numel],
                weight_type=self.so2_linear_type,
            )
        self.linear_down = uvSO2Linear(
            mmax,
            lmax,
            self.edge_ace_hidden
            if self.use_asymmetric_contraction
            else self.edge_wise_hidden,
            self.edge_wise_hidden,
            num_components_in=num_components_in,
            weight_type=self.so2_linear_type,
        )
        if self.use_radial_rotary_attention:
            self.query_proj = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel,
                self.edge_wise_hidden,
                weight_type=self.so2_linear_type,
            )
            self.key_proj = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel,
                self.edge_wise_hidden,
                weight_type=self.so2_linear_type,
            )
            if self.use_radial_phase:
                self.radial_proj = torchLinear(num_radial_basis, 2 * self.num_head)
            else:
                self.radial_proj = torchLinear(num_radial_basis, self.num_head)
            torch.nn.init.zeros_(self.radial_proj.weight)
            torch.nn.init.zeros_(self.radial_proj.bias)
            self.attention_scale = 1.0 / math.sqrt(
                self.num_channel_per_head * self.split_list[1]
            )
            self.graph_softmax = GraphSoftmax()
            if self.use_temperature:
                self.temperature_min = 0.25
                self.temperature_max = 4.0
                initial_temperature = 1.0
                initial_temperature_logit = math.log(
                    (initial_temperature - self.temperature_min)
                    / (self.temperature_max - initial_temperature)
                )
                self.temperature_logit = torch.nn.Parameter(
                    torch.full((self.num_head,), initial_temperature_logit)
                )

    def _complex_qk_attention(
        self, query: torch.Tensor, key: torch.Tensor, edge_feats: torch.Tensor
    ) -> torch.Tensor:

        B = query.size(0)
        H = self.num_head
        C = self.num_channel_per_head

        # radial bias and pahse
        if self.use_radial_phase:
            radial_proj = self.radial_proj(edge_feats)
            radial_bias = radial_proj[:, :H]
            radial_phase = math.pi * torch.tanh(radial_proj[:, H:])
        else:
            radial_bias = self.radial_proj(edge_feats)

        # m = 0
        n = self.lmax + 1
        query_m0 = query[:, :n].view(B, n, H, C)
        key_m0 = key[:, :n].view(B, n, H, C)
        score = (query_m0 * key_m0).sum(dim=(1, 3))

        # m > 0
        if self.use_radial_phase:
            offset = n
            for m in range(1, self.mmax + 1):
                n = self.lmax + 1 - m
                query_m = query[:, offset : offset + 2 * n].view(B, 2, n, H, C)
                key_m = key[:, offset : offset + 2 * n].view(B, 2, n, H, C)
                offset += 2 * n
                phase = (m * radial_phase).view(B, 1, H, 1)
                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                key_real = cos_phase * key_m[:, 0] - sin_phase * key_m[:, 1]
                key_imag = sin_phase * key_m[:, 0] + cos_phase * key_m[:, 1]
                score = score + (
                    query_m[:, 0] * key_real + query_m[:, 1] * key_imag
                ).sum(dim=(1, 3))
        else:
            offset = n
            for m in range(1, self.mmax + 1):
                n = self.lmax + 1 - m
                query_m = query[:, offset : offset + 2 * n].view(B, 2, n, H, C)
                key_m = key[:, offset : offset + 2 * n].view(B, 2, n, H, C)
                offset += 2 * n
                score = score + (
                    query_m[:, 0] * key_m[:, 0] + query_m[:, 1] * key_m[:, 1]
                ).sum(dim=(1, 3))

        if self.use_temperature:
            temperature = self.temperature_min + (
                self.temperature_max - self.temperature_min
            ) * torch.sigmoid(self.temperature_logit)
            return score * self.attention_scale * temperature + radial_bias

        return score * self.attention_scale + radial_bias

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        radial_basis: torch.Tensor,
    ) -> torch.Tensor:

        num_nodes = x.size(0)
        num_edges = w.size(0)
        x = self.reshape_in(x)
        m_ij = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        m_ij = torch.bmm(wigner, m_ij)

        if self.use_radial_rotary_attention:
            key = self.key_proj(m_ij[:, :, : self.num_channel])
            query = self.query_proj(m_ij[:, :, self.num_channel :])
            real_alpha = self._complex_qk_attention(query, key, radial_basis)

        w = w.view(num_edges, self.num_components, self.num_channel * 2)
        w = torch.index_select(w, dim=1, index=self.expand_index)
        m_ij = w * m_ij

        if self.use_asymmetric_contraction:
            coefs = self.nonlinearity.scalar_act(self.linear_coefs(m_ij).squeeze(-1))
            m_ij_2 = self.linear_glu(m_ij)
            m_ij = self.linear_up(m_ij)
            gate = m_ij.narrow(1, 0, self.split_list[0])
            m_ij = m_ij.narrow(1, self.split_list[0], self.split_list[1])
            m_ij = (
                m_ij + self.nonlinearity(m_ij, gate) + self.ece(m_ij, m_ij_2, coefs)
            )  # x + x**2 + x**3 TODO, forget scale
        else:
            m_ij = self.linear_up(m_ij)
            gate = m_ij.narrow(1, 0, self.split_list[0])
            m_ij = m_ij.narrow(1, self.split_list[0], self.split_list[1])
            m_ij = self.nonlinearity(m_ij, gate)

        m_ij = self.linear_down(m_ij)

        if self.use_radial_rotary_attention:
            real_alpha = self.graph_softmax(
                real_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff
            )  # [edge, head]
            if cutoff is not None:
                real_alpha = real_alpha * cutoff
            real_alpha = real_alpha.view(num_edges, 1, self.num_head, 1)
            m_ij = m_ij.view(
                num_edges, m_ij.size(1), self.num_head, self.num_channel_per_head
            )
            m_ij = real_alpha * m_ij
            m_ij = m_ij.view(num_edges, m_ij.size(1), self.edge_wise_hidden)
        else:
            if cutoff is not None:
                m_ij = m_ij * cutoff.unsqueeze(-1)

        m_ij = torch.bmm(wigner_inv, m_ij)

        return self.reshape_out.inverse(
            scatter_sum(
                m_ij,
                edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )
        )
