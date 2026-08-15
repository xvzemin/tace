################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch

from tace.utils.torch_scatter import scatter_sum
from ..layout import LayoutTransform
from ..linear import torchLinear
from ..o2.legacy import (
    SO2Gate,
    so2_expand_index,
    uvSO2Linear,
)
from ..softmax import GraphSoftmax
from .asymmetric_contraction import ComplexProductBasis


class OAM20270705(torch.nn.Module):
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
        use_so2_edge_ace: bool,
        use_graph_softmax: bool,
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
        self.use_graph_softmax = use_graph_softmax
        self.use_so2_edge_ace = use_so2_edge_ace
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        self.use_radial_phase = use_radial_phase

        self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
        self.weight_numel = self.num_components * self.num_channel * 2
        self.register_buffer("expand_index", expand_index, persistent=False)

        start_m = 0 if gate_m0 else 1
        if self.use_so2_edge_ace:
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
            self.edge_ace_hidden if self.use_so2_edge_ace else self.edge_wise_hidden,
            num_components_out=num_components_out,
            weight_type=self.so2_linear_type,
        )
        self.nonlinearity = SO2Gate(
            mmax,
            lmax,
            self.edge_ace_hidden if self.use_so2_edge_ace else self.edge_wise_hidden,
            channel_wise=self.use_so2_edge_ace,
            gate_m0=gate_m0,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
        )
        if self.use_so2_edge_ace:
            self.linear_glu = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel * 2,
                self.edge_ace_hidden
                if self.use_so2_edge_ace
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
            self.edge_ace_hidden if self.use_so2_edge_ace else self.edge_wise_hidden,
            self.edge_wise_hidden,
            num_components_in=num_components_in,
            weight_type=self.so2_linear_type,
        )
        if self.use_graph_softmax:
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

        if self.use_graph_softmax:
            key = self.key_proj(m_ij[:, :, : self.num_channel])
            query = self.query_proj(m_ij[:, :, self.num_channel :])
            real_alpha = self._complex_qk_attention(query, key, radial_basis)

        w = w.view(num_edges, self.num_components, self.num_channel * 2)
        w = torch.index_select(w, dim=1, index=self.expand_index)
        m_ij = w * m_ij

        if self.use_so2_edge_ace:
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

        if self.use_graph_softmax:
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