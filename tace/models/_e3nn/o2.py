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
        self.q_proj = o2.Linear(self.irreps, self.irreps, biases=True)
        self.k_proj = o2.Linear(self.irreps, self.irreps, biases=True)
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
    ece_path_mode: str = "expand"

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        mmax: int,
        act_0e: torch.nn.Module,
        act_0o: Optional[torch.nn.Module],
        act_lm: torch.nn.Module,
        correlation: int,
        num_head: int,
        num_radial_basis: int,
        use_asymmetric_contraction: bool,
        use_radial_rotary_attention: bool,
        edge_ace_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channel = num_channel
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        self.lmax = lmax
        self.mmax = min(self.irreps_in.lmax, mmax)
        self.correlation = correlation
        self.num_head = num_head
        if not (
            o2.Irreps.common_multiplicity(self.irreps_in)
            == o2.Irreps.common_multiplicity(self.irreps_out)
            == num_channel
        ):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")

        self.local_frame_in = o2.LocalFrame(self.irreps_in, lmax, self.mmax)
        self.local_frame_out = o2.LocalFrame(self.irreps_out, lmax, self.mmax)
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
        self.use_asymmetric_contraction = (
            use_asymmetric_contraction and self.irreps_in.lmax > 0
        )

        if self.use_asymmetric_contraction:
            self.nonlinearity = None
            contraction_irreps = o2.Irreps(
                [(ir, self.edge_ace_hidden) for ir, _ in self.local_irreps_out]
            )
            self.asymmetric_contraction = o2.AsymmetricContraction(
                contraction_irreps,
                contraction_irreps,
                correlation,
                algorithm="edge",
                path_mode=self.ece_path_mode,
            )
            projection_irreps = o2.Irreps(
                [
                    (ir, correlation * self.edge_ace_hidden)
                    for ir, _ in contraction_irreps
                ]
                + [(o2.Irrep("0e"), self.asymmetric_contraction.weight_numel)]
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                projection_irreps,
                biases=True,
            )
        else:
            self.asymmetric_contraction = None
            hidden_irreps = self.local_irreps_out.filter(
                keep=lambda ir_mul: self.local_irreps_in.count(ir_mul.ir) > 0
            )
            scalar_entries = []
            scalar_acts = []
            gated_entries = []
            for ir, mul in hidden_irreps:
                if ir.is_even_scalar():
                    scalar_entries.append((ir, mul))
                    scalar_acts.append(act_0e)
                elif ir.is_odd_scalar() and act_0o is not None:
                    scalar_entries.append((ir, mul))
                    scalar_acts.append(act_0o)
                else:
                    gated_entries.append((ir, mul))
            irreps_gated = o2.Irreps(gated_entries)
            irreps_gates = (
                o2.Irreps([(o2.Irrep("0e"), irreps_gated.num_irreps)])
                if irreps_gated.num_irreps
                else o2.Irreps()
            )
            self.nonlinearity = o2.Gate(
                o2.Irreps(scalar_entries),
                scalar_acts,
                irreps_gates,
                [act_lm] if len(irreps_gates) else [],
                irreps_gated,
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                self.nonlinearity.irreps_in,
                biases=True,
            )

        hidden_irreps = (
            self.asymmetric_contraction.irreps_out
            if self.asymmetric_contraction is not None
            else self.nonlinearity.irreps_out
        )
        self.linear_down = o2.Linear(
            hidden_irreps,
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

        if self.asymmetric_contraction is not None:
            contraction_inputs = [[] for _ in range(self.correlation)]
            projected_slices = self.linear_up.irreps_out.slices()
            contraction_entries = len(self.asymmetric_contraction.irreps_in)
            for index, ((ir, _), ir_slice) in enumerate(
                zip(self.linear_up.irreps_out, projected_slices)
            ):
                if index == contraction_entries:
                    contraction_weights = projected[..., ir_slice]
                    break
                values = projected[..., ir_slice].reshape(
                    projected.size(0),
                    ir.dim,
                    self.correlation,
                    self.edge_ace_hidden,
                )
                for order in range(self.correlation):
                    contraction_inputs[order].append(
                        values[..., order, :].reshape(
                            projected.size(0), ir.dim * self.edge_ace_hidden
                        )
                    )
            else:
                raise RuntimeError("Missing asymmetric-contraction weights.")
            hidden = self.asymmetric_contraction(
                [torch.cat(values, dim=-1) for values in contraction_inputs],
                contraction_weights,
            )
        else:
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
    ece_path_mode: str = "expand"

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_irreps: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        mmax: int,
        act_0e: torch.nn.Module,
        act_0o: Optional[torch.nn.Module],
        act_lm: torch.nn.Module,
        correlation: int,
        num_head: int,
        num_radial_basis: int,
        use_asymmetric_contraction: bool,
        use_radial_rotary_attention: bool,
        edge_ace_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_irreps = o3.Irreps(magnetic_irreps)
        self.num_channel = num_channel
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        self.lmax = lmax
        self.mmax = min(max(self.irreps_in.lmax, self.magnetic_irreps.lmax), mmax)
        self.correlation = correlation
        self.num_head = num_head
        if not (
            o2.Irreps.common_multiplicity(self.irreps_in)
            == o2.Irreps.common_multiplicity(self.irreps_out)
            == num_channel
        ):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")

        self.local_frame_in = o2.LocalFrame(self.irreps_in, lmax, self.mmax)
        self.local_frame_out = o2.LocalFrame(self.irreps_out, lmax, self.mmax)
        self.magnetic_frame = o2.LocalFrame(self.magnetic_irreps, lmax, self.mmax)
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
            self.magnetic_irreps,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.node_irreps = self.local_frame_in.irreps_out
        self.local_magnetic_irreps = o2.Irreps(
            [(ir, mul * num_channel) for ir, mul in self.magnetic_frame.irreps_out]
        )
        self.local_irreps_in = (
            self.node_irreps
            + self.node_irreps
            + self.local_magnetic_irreps
            + self.local_magnetic_irreps
        ).regroup()
        self.local_irreps_out = self.local_frame_out.irreps_out
        self.use_asymmetric_contraction = (
            use_asymmetric_contraction and self.irreps_in.lmax > 0
        )

        if self.use_asymmetric_contraction:
            self.nonlinearity = None
            contraction_irreps = o2.Irreps(
                [(ir, self.edge_ace_hidden) for ir, _ in self.local_irreps_out]
            )
            self.asymmetric_contraction = o2.AsymmetricContraction(
                contraction_irreps,
                contraction_irreps,
                correlation,
                algorithm="edge",
                path_mode=self.ece_path_mode,
            )
            projection_irreps = o2.Irreps(
                [
                    (ir, correlation * self.edge_ace_hidden)
                    for ir, _ in contraction_irreps
                ]
                + [(o2.Irrep("0e"), self.asymmetric_contraction.weight_numel)]
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                projection_irreps,
                biases=True,
            )
        else:
            self.asymmetric_contraction = None
            hidden_irreps = self.local_irreps_out.filter(
                keep=lambda ir_mul: self.local_irreps_in.count(ir_mul.ir) > 0
            )
            scalar_entries = []
            scalar_acts = []
            gated_entries = []
            for ir, mul in hidden_irreps:
                if ir.is_even_scalar():
                    scalar_entries.append((ir, mul))
                    scalar_acts.append(act_0e)
                elif ir.is_odd_scalar() and act_0o is not None:
                    scalar_entries.append((ir, mul))
                    scalar_acts.append(act_0o)
                else:
                    gated_entries.append((ir, mul))
            irreps_gated = o2.Irreps(gated_entries)
            irreps_gates = (
                o2.Irreps([(o2.Irrep("0e"), irreps_gated.num_irreps)])
                if irreps_gated.num_irreps
                else o2.Irreps()
            )
            self.nonlinearity = o2.Gate(
                o2.Irreps(scalar_entries),
                scalar_acts,
                irreps_gates,
                [act_lm] if len(irreps_gates) else [],
                irreps_gated,
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                self.nonlinearity.irreps_in,
                biases=True,
            )

        hidden_irreps = (
            self.asymmetric_contraction.irreps_out
            if self.asymmetric_contraction is not None
            else self.nonlinearity.irreps_out
        )
        self.linear_down = o2.Linear(
            hidden_irreps,
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

    def _to_local(
        self,
        node_features: torch.Tensor,
        magnetic_node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.reshape_in(node_features)
        magnetic_node_attrs = self.reshape_magnetic(magnetic_node_attrs)
        paired_node = self.local_frame_in.to_local(node_features[edge_index.T], wigner)
        paired_magnetic = self.magnetic_frame.to_local(
            magnetic_node_attrs[edge_index.T], wigner
        )
        magnetic = []
        for (ir, mul), ir_slice in zip(
            self.magnetic_frame.irreps_out,
            self.magnetic_frame.irreps_out.slices(),
        ):
            values = paired_magnetic[..., ir_slice].reshape(
                paired_magnetic.size(0), 2, ir.dim, mul
            )
            values = values.repeat_interleave(self.num_channel, dim=-1)
            magnetic.append(
                values.reshape(
                    paired_magnetic.size(0),
                    2,
                    ir.dim * mul * self.num_channel,
                )
            )
        paired_magnetic = torch.cat(magnetic, dim=-1)
        return (
            node_features,
            paired_node[:, 0],
            paired_node[:, 1],
            paired_magnetic[:, 0],
            paired_magnetic[:, 1],
        )

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_magnetic: torch.Tensor,
        target_magnetic: torch.Tensor,
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
            (target_magnetic, self.local_magnetic_irreps),
            (source_magnetic, self.local_magnetic_irreps),
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

        if self.asymmetric_contraction is not None:
            contraction_inputs = [[] for _ in range(self.correlation)]
            projected_slices = self.linear_up.irreps_out.slices()
            contraction_entries = len(self.asymmetric_contraction.irreps_in)
            for index, ((ir, _), ir_slice) in enumerate(
                zip(self.linear_up.irreps_out, projected_slices)
            ):
                if index == contraction_entries:
                    contraction_weights = projected[..., ir_slice]
                    break
                values = projected[..., ir_slice].reshape(
                    projected.size(0),
                    ir.dim,
                    self.correlation,
                    self.edge_ace_hidden,
                )
                for order in range(self.correlation):
                    contraction_inputs[order].append(
                        values[..., order, :].reshape(
                            projected.size(0), ir.dim * self.edge_ace_hidden
                        )
                    )
            else:
                raise RuntimeError("Missing asymmetric-contraction weights.")
            hidden = self.asymmetric_contraction(
                [torch.cat(values, dim=-1) for values in contraction_inputs],
                contraction_weights,
            )
        else:
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
        magnetic_node_attrs: torch.Tensor,
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
            source_magnetic,
            target_magnetic,
        ) = self._to_local(node_feats, magnetic_node_attrs, edge_index, wigner)
        message = self._convolution(
            source_features,
            target_features,
            source_magnetic,
            target_magnetic,
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
