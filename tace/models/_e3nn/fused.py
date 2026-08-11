################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import math
from typing import Union

import torch
from e3nn import o3

from tace.utils.env import (
    acceleration_enabled,
)
from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..legacy_so2 import (
    SO2Gate,
    so2_expand_index,
    so3_expand_index,
    uuSO2Linear,
    uvSO2Linear,
)
from ..linear import torchLinear
from ..softmax import GraphSoftmax
from .asymmetric_contraction import ComplexProductBasis, SO2ASymmetricContraction
from .paths import generate_paths


class uuuTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
        trainable: bool = False,
        warning: bool = False,
        identical_inputs: bool = False,
    ) -> None:
        super().__init__()

        instructions, actual_irreps_out = generate_paths(
            irreps_out=irreps_out,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            l1l2=l1l2,
            l2l3=l2l3,
            l3l1=l3l1,
            e3nn_mode="uuu",
            trainable=trainable,
            identical_inputs=identical_inputs,
        )

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = actual_irreps_out
        self.instructions = instructions
        self.weight_numel = self.tp.weight_numel
        self.use_eqt = acceleration_enabled("eqt")
        self.use_cue = acceleration_enabled("cue")

        if self.use_eqt:
            from ..eqt import e3nnEqtTensorProduct

            self.fused_tp = e3nnEqtTensorProduct(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=actual_irreps_out,
                num_channel=irreps_in2.count("1o"),
                path=instructions,
                trainable=trainable,
            )
        # elif self.use_cue and not trainable:
        #     from ..cue import e3nnCueTensorProduct
        #     self.fused_tp = e3nnCueTensorProduct(
        #         irreps_in1=irreps_in1,
        #         irreps_in2=irreps_in2,
        #         irreps_out=irreps_out,
        #         l1l2=l1l2,
        #         l2l3=l2l3,
        #         l3l1=l3l1,
        #         trainable=trainable,
        #     )
        elif warning:
            logging.warning(
                "Correlation >= 3 is running without Equitorch. "
                "For acceleration options, see "
                "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
            )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, ws: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, ws)
        return self.tp(x, y, ws)


class uvuTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple],
        *,
        shared_weights: bool,
    ) -> None:
        super().__init__()

        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)
        if any(instruction[3] != "uvu" for instruction in instructions):
            raise ValueError("uvuTensorProduct only accepts uvu instructions")

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=shared_weights,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.instructions = self.tp.instructions
        self.weight_numel = self.tp.weight_numel
        self.shared_weights = shared_weights
        use_oeq = acceleration_enabled("oeq")
        oeq_compatible = all(multiplicity == 1 for multiplicity, _ in irreps_in2)
        oeq_compatible = oeq_compatible and all(
            instruction[4] for instruction in instructions
        )
        self.use_oeq = use_oeq and oeq_compatible

        if use_oeq and not oeq_compatible:
            logging.warning(
                "OEQ uvu tensor products require weighted instructions and "
                "multiplicity-one irreps_in2 for e3nn-compatible weight ordering. "
                "Falling back to e3nn."
            )

        if self.use_oeq:
            from ..oeq import e3nnOeqTensorProduct

            self.fused_tp = e3nnOeqTensorProduct(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=irreps_out,
                instructions=instructions,
                shared_weights=shared_weights,
            )
            if self.fused_tp.weight_numel != self.weight_numel:
                raise RuntimeError(
                    "OEQ and e3nn generated different uvu tensor-product paths: "
                    f"{self.fused_tp.weight_numel} != {self.weight_numel}."
                )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, weights)
        return self.tp(x, y, weights)


class O3ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
        instructions: Union[list[tuple], None] = None,
    ) -> None:
        super().__init__()

        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        explicit_instructions = instructions is not None
        if instructions is None:
            instructions, actual_irreps_out = generate_paths(
                irreps_out=irreps_out,
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
                e3nn_mode="uvu",
            )
        else:
            actual_irreps_out = irreps_out

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = actual_irreps_out
        self.instructions = instructions
        self.weight_numel = self.tp.weight_numel

        self.use_oeq = acceleration_enabled("oeq")
        self.use_cue = acceleration_enabled("cue")
        self.use_aoti = acceleration_enabled("compile")

        if self.use_aoti and self.use_cue:
            logging.warning(
                "CUE and AOTI cannot be used simultaneously in Scatter Tensor Product. "
                "Falling back to AOTI with OEQ instead. "
                "If execution fails, install OpenEquivariance with: pip install openequivariance"
            )
            self.use_oeq = True
            self.use_cue = False
        else:
            pass

        if self.use_oeq:
            from ..oeq import e3nnOeqScatterTensorProduct

            self.fused_tp = e3nnOeqScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=self.instructions,
            )
        elif self.use_cue and not explicit_instructions:
            from ..cue import e3nnCueScatterTensorProduct

            self.fused_tp = e3nnCueScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
            )
        elif self.use_cue:
            logging.warning(
                "CUE scatter tensor products do not support explicit instructions. "
                "Falling back to e3nn for this tensor product."
            )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, w, edge_index)
        return scatter_sum(
            self.tp(x[edge_index[0]], y, w), edge_index[1], dim=0, dim_size=x.size(0)
        )


class uuSO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        l1l3: Union[str, None],
        weight_type: str,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
    ) -> None:
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        self.weight_type = weight_type
        self.path_mode = "sum"
        self.l1l3 = l1l3
        self.linear_up = uuSO2Linear(
            self.mmax,
            self.lmax,
            self.num_channel,
            weight_type=self.weight_type,
            path_mode=self.path_mode,
            path_norm=self.path_mode == "sum",
            l1l3=self.l1l3,
        )
        self.weight_numel = self.linear_up.weight_numel
        if acceleration_enabled("eqx"):
            # Temporary bridge until this operator is provided by eqx itself.
            from ..triton_ops import UUSO2Scatter

            self.fused_tp = UUSO2Scatter(self.linear_up)
        else:
            logging.warning(
                "You are not using the EQX preview operator; export TACE_USE_EQX=1. "
                "For acceleration options, see "
                "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
            )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:

        num_nodes = x.size(0)
        x = self.reshape_in(x)

        if hasattr(self, "fused_tp"):
            return self.reshape_out.inverse(
                self.fused_tp(
                    x,
                    w,
                    edge_index,
                    wigner,
                    wigner_inv,
                )
            )

        m_ij = x[edge_index[0]]
        m_ij = torch.bmm(wigner, m_ij)
        m_ij = self.linear_up(m_ij, w)
        m_ij = torch.bmm(wigner_inv, m_ij)

        return self.reshape_out.inverse(
            scatter_sum(
                m_ij,
                edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )
        )


# A little BUG
class uvSO2TensorProduct(torch.nn.Module):
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
