################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import math
from typing import Optional

import torch
from e3nn import o3

from eqx import o2
from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..linear import torchLinear
from ..softmax import GraphSoftmax
from ..utils import repr_without


class RadialRotaryComplexAttention(torch.nn.Module):
    def __init__(
        self,
        irreps: o2.Irreps,
        message_irreps: o2.Irreps,
        num_head: int,
        num_radial_basis: int,
    ) -> None:
        super().__init__()
        self.irreps = o2.Irreps(irreps)
        self.message_irreps = o2.Irreps(message_irreps)
        self.num_head = num_head
        for _, mul in self.irreps:
            if num_head < 1 or mul % num_head != 0:
                raise ValueError("num_head must divide every O2 multiplicity.")
        for _, mul in self.message_irreps:
            if mul % num_head != 0:
                raise ValueError("num_head must divide every message multiplicity.")
        self.q_proj = o2.Linear(self.irreps, self.irreps)
        self.k_proj = o2.Linear(self.irreps, self.irreps)
        self.radial_proj = torchLinear(num_radial_basis, 2 * num_head)
        torch.nn.init.zeros_(self.radial_proj.weight)
        torch.nn.init.zeros_(self.radial_proj.bias)
        self.scale = math.sqrt(num_head / self.irreps.dim)
        self.graph_softmax = GraphSoftmax()

    def forward(
        self,
        message: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        query = self.q_proj(target_features)
        key = self.k_proj(source_features)
        score = edge_radial_basis.new_zeros(
            edge_radial_basis.size(0),
            self.num_head,
        )
        for (ir, mul), ir_slice in zip(self.irreps, self.irreps.slices()):
            shape = (query.size(0), ir.dim, self.num_head, mul // self.num_head)
            score = score + (
                query[..., ir_slice].reshape(shape) * key[..., ir_slice].reshape(shape)
            ).sum(dim=(1, 3))
        radial_scale, radial_shift = self.radial_proj(edge_radial_basis).chunk(
            2, dim=-1
        )
        score = score * self.scale * (2.0 * torch.sigmoid(radial_scale)) + radial_shift
        attention = (
            self.graph_softmax(
                score,
                edge_index[1],
                num_nodes=num_nodes,
                exp_rescale=edge_cutoff,
            )
            * edge_cutoff
        )

        outputs = []
        for (ir, mul), ir_slice in zip(
            self.message_irreps,
            self.message_irreps.slices(),
        ):
            shape = (
                message.size(0),
                ir.dim,
                self.num_head,
                mul // self.num_head,
            )
            output = message[..., ir_slice].reshape(shape)
            output = output * attention[:, None, :, None]
            outputs.append(output.reshape(message.size(0), ir.dim * mul))
        return torch.cat(outputs, dim=-1)


class O2ScatterTensorProduct(torch.nn.Module):

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        mmax: int,
        even_scalar_act: torch.nn.Module,
        odd_scalar_act: Optional[torch.nn.Module],
        tensor_act: torch.nn.Module,
        num_head: int,
        num_radial_basis: int,
        use_radial_rotary_attention: bool,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channel = num_channel
        self.lmax = lmax
        self.mmax = min(self.irreps_in.lmax, mmax)
        self.num_head = num_head
        if not (
            o2.Irreps.common_multiplicity(self.irreps_in)
            == o2.Irreps.common_multiplicity(self.irreps_out)
            == num_channel
        ):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")
        self.local_frame_in = o2.LocalFrame(self.irreps_in, lmax, self.mmax)
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            lmax,
            self.mmax,
            reverse=True,
        )
        self.reshape_in = LayoutTransform(
            self.irreps_in,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_out = LayoutTransform(
            self.irreps_out,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.node_irreps = self.local_frame_in.irreps_out
        self.local_irreps_in = 2 * self.node_irreps
        self.local_irreps_out = self.local_frame_out.irreps_out
        hidden_irreps = self.local_irreps_out.filter(
            keep=lambda ir_mul: self.local_irreps_in.count(ir_mul.ir) > 0
        )
        scalar_entries = []
        scalar_acts = []
        gated_entries = []
        for ir, mul in hidden_irreps:
            if ir.is_invariant_scalar():
                scalar_entries.append((ir, mul))
                scalar_acts.append(even_scalar_act)
            elif ir.m == 0 and odd_scalar_act is not None:
                scalar_entries.append((ir, mul))
                scalar_acts.append(odd_scalar_act)
            else:
                gated_entries.append((ir, mul))
        irreps_gated = o2.Irreps(gated_entries)
        irreps_gates = (
            o2.Irreps([(o2.Irrep("0ee"), irreps_gated.num_irreps)])
            if irreps_gated.num_irreps
            else o2.Irreps()
        )
        self.nonlinearity = o2.Gate(
            o2.Irreps(scalar_entries),
            scalar_acts,
            irreps_gates,
            [tensor_act] if len(irreps_gates) else [],
            irreps_gated,
        )
        self.linear_up = o2.Linear(
            self.local_irreps_in,
            self.nonlinearity.irreps_in,
        )
        self.linear_down = o2.Linear(
            self.nonlinearity.irreps_out,
            self.local_irreps_out,
        )
        self.weight_numel = self.local_irreps_in.num_irreps

        self.use_radial_rotary_attention = (
            use_radial_rotary_attention and self.node_irreps.mmax > 0
        )
        self.attention = (
            RadialRotaryComplexAttention(
                self.node_irreps,
                self.local_irreps_out,
                num_head,
                num_radial_basis,
            )
            if self.use_radial_rotary_attention
            else None
        )

    def __repr__(self) -> str:
        return repr_without(self, "reshape_in", "reshape_out")

    def _to_local(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.reshape_in(node_features)
        paired = self.local_frame_in.to_local(
            node_features[edge_index.T],
            wigner,
        )
        return node_features, paired[:, 0], paired[:, 1]

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Optional[torch.Tensor],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        weighted = []
        offset = 0
        for (ir, mul), ir_slice in zip(
            self.node_irreps,
            self.node_irreps.slices(),
        ):
            values = torch.cat(
                (
                    target_features[..., ir_slice].view(
                        target_features.size(0), ir.dim, mul
                    ),
                    source_features[..., ir_slice].view(
                        source_features.size(0), ir.dim, mul
                    ),
                ),
                dim=-1,
            )
            width = 2 * mul
            weight = conv_weights[..., offset : offset + width].unsqueeze(-2)
            weighted.append((values * weight).reshape(values.size(0), ir.dim * width))
            offset += width
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")
        projected = self.linear_up(torch.cat(weighted, dim=-1))
        hidden = self.nonlinearity(projected)
        message = self.linear_down(hidden)
        if self.attention is not None:
            if edge_radial_basis is None:
                raise ValueError(
                    "O2 radial rotary attention requires edge_radial_basis."
                )
            message = self.attention(
                message,
                source_features,
                target_features,
                edge_radial_basis,
                edge_index,
                edge_cutoff,
                num_nodes,
            )
        return message

    def _to_global(
        self,
        message: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        message = self.local_frame_out.to_global(message, wigner_inv)
        if self.attention is None:
            message = message * edge_cutoff
        message = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return self.reshape_out.inverse(message)

    def forward(
        self,
        node_feats: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Optional[torch.Tensor],
        wigner_inv: Optional[torch.Tensor],
        edge_radial_basis: Optional[torch.Tensor] = None,
        edge_cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        node_features, source_features, target_features = self._to_local(
            node_feats, edge_index, wigner
        )
        message = self._convolution(
            source_features,
            target_features,
            conv_weights,
            edge_index,
            edge_radial_basis,
            edge_cutoff,
            node_features.size(0),
        )
        return self._to_global(
            message,
            edge_index,
            wigner_inv,
            edge_cutoff,
            node_features.size(0),
        )


class O2ScatterMagneticTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_edge_irreps: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        mmax: int,
        even_scalar_act: torch.nn.Module,
        odd_scalar_act: Optional[torch.nn.Module],
        tensor_act: torch.nn.Module,
        num_head: int,
        num_radial_basis: int,
        use_radial_rotary_attention: bool,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_edge_irreps = o3.Irreps(magnetic_edge_irreps)
        self.num_channel = num_channel
        self.lmax = lmax
        self.mmax = min(
            max(self.irreps_in.lmax, self.magnetic_edge_irreps.lmax),
            mmax,
        )
        self.num_head = num_head
        if not (
            o2.Irreps.common_multiplicity(self.irreps_in)
            == o2.Irreps.common_multiplicity(self.irreps_out)
            == num_channel
        ):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")
        if (
            o2.Irreps.common_multiplicity(self.magnetic_edge_irreps)
            != num_channel
        ):
            raise ValueError(
                "magnetic_edge_irreps multiplicity must equal num_channel."
            )

        self.local_frame_in = o2.LocalFrame(self.irreps_in, lmax, self.mmax)
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            lmax,
            self.mmax,
            reverse=True,
        )
        self.magnetic_frame = o2.LocalFrame(
            self.magnetic_edge_irreps,
            lmax,
            self.mmax,
        )
        self.reshape_in = LayoutTransform(
            self.irreps_in,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_out = LayoutTransform(
            self.irreps_out,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_magnetic = LayoutTransform(
            self.magnetic_edge_irreps,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.node_irreps = self.local_frame_in.irreps_out
        self.local_magnetic_irreps = self.magnetic_frame.irreps_out
        self.local_irreps_in = (
            self.node_irreps
            + self.node_irreps
            + self.local_magnetic_irreps
        ).regroup()
        self.local_irreps_out = self.local_frame_out.irreps_out
        self.use_time_reversal = any(
            ir.t == -1 for ir, _ in self.local_irreps_in
        )
        hidden_irreps = self.local_irreps_out.filter(
            keep=lambda ir_mul: self.local_irreps_in.count(ir_mul.ir) > 0
        )
        scalar_entries = []
        scalar_acts = []
        gated_entries = []
        for ir, mul in hidden_irreps:
            if ir.is_invariant_scalar():
                scalar_entries.append((ir, mul))
                scalar_acts.append(even_scalar_act)
            elif ir.m == 0 and odd_scalar_act is not None:
                scalar_entries.append((ir, mul))
                scalar_acts.append(odd_scalar_act)
            else:
                gated_entries.append((ir, mul))

        irreps_gated = o2.Irreps(gated_entries)
        if self.use_time_reversal:
            gate_entries = [(o2.Irrep("0ee"), mul) for _, mul in gated_entries]
            gate_acts = [tensor_act] * len(gate_entries)
            time_odd_scalars = [
                ir for ir, _ in self.local_irreps_in if ir.m == 0 and ir.t == -1
            ]
            time_odd_irreps = [
                ir for ir, _ in self.local_irreps_in if ir.t == -1
            ]
            for ir_out, mul in self.local_irreps_out:
                path = next(
                    (
                        (ir_gate, ir_gated)
                        for ir_gate in time_odd_scalars
                        for ir_gated in time_odd_irreps
                        if ir_gated * ir_gate == (ir_out,)
                    ),
                    None,
                )
                if path is not None:
                    ir_gate, ir_gated = path
                    gate_entries.append((ir_gate, mul))
                    gate_acts.append(odd_scalar_act)
                    gated_entries.append((ir_gated, mul))
            irreps_gated = o2.Irreps(gated_entries)
            irreps_gates = o2.Irreps(gate_entries)
        else:
            irreps_gates = (
                o2.Irreps([(o2.Irrep("0ee"), irreps_gated.num_irreps)])
                if irreps_gated.num_irreps
                else o2.Irreps()
            )
            gate_acts = [tensor_act] if len(irreps_gates) else []
        self.nonlinearity = o2.Gate(
            o2.Irreps(scalar_entries),
            scalar_acts,
            irreps_gates,
            gate_acts,
            irreps_gated,
        )
        self.linear_up = o2.Linear(
            self.local_irreps_in,
            self.nonlinearity.irreps_in,
            biases=True,
        )
        self.linear_down = o2.Linear(
            self.nonlinearity.irreps_out,
            self.local_irreps_out,
            biases=True,
        )
        self.weight_numel = self.local_irreps_in.num_irreps
        self.use_radial_rotary_attention = (
            use_radial_rotary_attention and self.node_irreps.mmax > 0
        )
        self.attention = (
            RadialRotaryComplexAttention(
                self.node_irreps,
                self.local_irreps_out,
                num_head,
                num_radial_basis,
            )
            if self.use_radial_rotary_attention
            else None
        )

    def __repr__(self) -> str:
        return repr_without(self, "reshape_in", "reshape_out")

    def _to_local(
        self,
        node_features: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.reshape_in(node_features)
        magnetic_edge_attrs = self.reshape_magnetic(magnetic_edge_attrs)
        paired_node = self.local_frame_in.to_local(node_features[edge_index.T], wigner)
        magnetic_edge_attrs = self.magnetic_frame.to_local(
            magnetic_edge_attrs,
            wigner,
        )
        return (
            node_features,
            paired_node[:, 0],
            paired_node[:, 1],
            magnetic_edge_attrs,
        )

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Optional[torch.Tensor],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        weighted = []
        offset = 0
        feature_sets = (
            (target_features, self.node_irreps),
            (source_features, self.node_irreps),
            (magnetic_edge_attrs, self.local_magnetic_irreps),
        )
        for ir, mul in self.local_irreps_in:
            inputs = []
            for features, irreps in feature_sets:
                for (input_ir, input_mul), ir_slice in zip(
                    irreps,
                    irreps.slices(),
                ):
                    if input_ir == ir:
                        inputs.append(
                            features[..., ir_slice].view(
                                features.size(0), ir.dim, input_mul
                            )
                        )
            values = torch.cat(inputs, dim=-1)
            weight = conv_weights[..., offset : offset + mul].unsqueeze(-2)
            weighted.append((values * weight).reshape(values.size(0), ir.dim * mul))
            offset += mul
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")
        projected = self.linear_up(torch.cat(weighted, dim=-1))
        hidden = self.nonlinearity(projected)
        message = self.linear_down(hidden)
        if self.attention is not None:
            if edge_radial_basis is None:
                raise ValueError(
                    "O2 radial rotary attention requires edge_radial_basis."
                )
            message = self.attention(
                message,
                source_features,
                target_features,
                edge_radial_basis,
                edge_index,
                edge_cutoff,
                num_nodes,
            )
        return message

    def _to_global(
        self,
        message: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        message = self.local_frame_out.to_global(message, wigner_inv)
        if self.attention is None:
            message = message * edge_cutoff
        message = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return self.reshape_out.inverse(message)

    def forward(
        self,
        node_feats: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Optional[torch.Tensor],
        wigner_inv: Optional[torch.Tensor],
        edge_radial_basis: Optional[torch.Tensor] = None,
        edge_cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        if wigner is None or wigner_inv is None:
            raise ValueError("O2 convolution requires Wigner matrices.")
        (
            node_features,
            source_features,
            target_features,
            magnetic_edge_attrs,
        ) = self._to_local(node_feats, magnetic_edge_attrs, edge_index, wigner)
        message = self._convolution(
            source_features,
            target_features,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            edge_radial_basis,
            edge_cutoff,
            node_features.size(0),
        )
        return self._to_global(
            message,
            edge_index,
            wigner_inv,
            edge_cutoff,
            node_features.size(0),
        )
