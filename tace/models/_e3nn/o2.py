################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch
from e3nn import o3

from eqx import o2

from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..linear import torchLinear
from ..softmax import GraphSoftmax


def _to_contraction_layout(
    features: tuple[torch.Tensor, ...],
    irreps: o2.Irreps,
    channels: int,
) -> torch.Tensor:
    if len(features) != len(irreps):
        raise ValueError("Expected one tensor per O(2) irrep group.")
    outputs = []
    for values, (mul, ir) in zip(features, irreps):
        leading_shape = values.shape[:-2]
        output = values.reshape(
            *leading_shape,
            ir.dim,
            mul,
            channels,
        )
        output = output.permute(
            *range(len(leading_shape)),
            len(leading_shape) + 1,
            len(leading_shape),
            len(leading_shape) + 2,
        )
        outputs.append(
            output.reshape(*leading_shape, mul * ir.dim, channels)
        )
    return torch.cat(outputs, dim=-2)


def _from_contraction_layout(
    features: torch.Tensor,
    irreps: o2.Irreps,
    channels: int,
) -> tuple[torch.Tensor, ...]:
    leading_shape = features.shape[:-2]
    outputs = []
    for (mul, ir), ir_slice in zip(irreps, irreps.slices()):
        values = features[..., ir_slice, :].reshape(
            *leading_shape,
            mul,
            ir.dim,
            channels,
        )
        values = values.permute(
            *range(len(leading_shape)),
            len(leading_shape) + 1,
            len(leading_shape),
            len(leading_shape) + 2,
        )
        outputs.append(
            values.reshape(*leading_shape, ir.dim, mul * channels)
        )
    if irreps.dim != features.size(-2):
        raise ValueError("Invalid contraction O(2) representation size.")
    return tuple(outputs)


class RadialRotaryComplexAttention(torch.nn.Module):
    def __init__(
        self,
        irreps: o2.Irreps,
        message_irreps: o2.Irreps,
        channels: int,
        num_head: int,
        num_radial_basis: int,
    ) -> None:
        super().__init__()
        if num_head < 1 or channels % num_head != 0:
            raise ValueError("num_head must divide channels.")
        self.irreps = o2.Irreps(irreps)
        self.message_irreps = o2.Irreps(message_irreps)
        self.channels = channels
        self.num_head = num_head
        self.channels_per_head = channels // num_head
        self.q_proj = o2.Linear(
            self.irreps,
            self.irreps,
            channels,
            channels,
            path_mode="uv",
            bias=True,
        )
        self.k_proj = o2.Linear(
            self.irreps,
            self.irreps,
            channels,
            channels,
            path_mode="uv",
            bias=True,
        )
        self.radial_proj = torchLinear(
            num_radial_basis,
            2 * num_head,
            bias=True,
        )
        torch.nn.init.zeros_(self.radial_proj.weight)
        torch.nn.init.zeros_(self.radial_proj.bias)
        components_per_head = self.irreps.dim * channels // num_head
        self.scale = 1.0 / math.sqrt(components_per_head)
        self.graph_softmax = GraphSoftmax()

    def _project_query_key(
        self,
        source_features: tuple[torch.Tensor, ...],
        target_features: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if len(source_features) != len(self.irreps) or len(target_features) != len(
            self.irreps
        ):
            raise ValueError("Expected one source and target tensor per O2 group.")
        query_features = self.q_proj.forward_grouped(target_features)
        key_features = self.k_proj.forward_grouped(source_features)
        return query_features, key_features

    def forward(
        self,
        message: tuple[torch.Tensor, ...],
        source_features: tuple[torch.Tensor, ...],
        target_features: tuple[torch.Tensor, ...],
        edge_radial_basis: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, ...]:
        if len(message) != len(self.message_irreps):
            raise ValueError("Expected one message tensor per output O2 group.")
        query_features, key_features = self._project_query_key(
            source_features,
            target_features,
        )
        score = edge_radial_basis.new_zeros(
            edge_radial_basis.size(0),
            self.num_head,
        )
        for query, key, (multiplicity, irrep) in zip(
            query_features,
            key_features,
            self.irreps,
        ):
            shape = (
                query.size(0),
                irrep.dim,
                multiplicity,
                self.num_head,
                self.channels_per_head,
            )
            score = score + (query.reshape(shape) * key.reshape(shape)).sum(
                dim=(1, 2, 4)
            )
        radial_scale, radial_shift = self.radial_proj(
            edge_radial_basis
        ).chunk(2, dim=-1)
        radial_scale = 2.0 * torch.sigmoid(radial_scale)
        score = score * self.scale * radial_scale + radial_shift

        attention = self.graph_softmax(
            score,
            edge_index[1],
            num_nodes=num_nodes,
            exp_rescale=edge_cutoff,
        )
        attention = attention * edge_cutoff

        outputs = []
        for features, (multiplicity, irrep) in zip(
            message,
            self.message_irreps,
        ):
            shape = (
                features.size(0),
                irrep.dim,
                multiplicity,
                self.num_head,
                self.channels_per_head,
            )
            output = features.reshape(shape) * attention[:, None, None, :, None]
            outputs.append(output.reshape_as(features))
        return tuple(outputs)


class O2ScatterTensorProduct(torch.nn.Module):

    ece_path_mode: str = "expand" # [expand, sum]

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        mmax: int,
        act_0e: torch.nn.Module,
        act_0o: Union[torch.nn.Module, None],
        act_lm: torch.nn.Module,
        correlation: int,
        num_head: int,
        num_radial_basis: int,
        use_asymmetric_contraction: bool,
        use_radial_rotary_attention: bool,
        edge_ace_hidden: Union[int, None] = None,
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
        self.use_asymmetric_contraction = use_asymmetric_contraction
        self.use_radial_rotary_attention = use_radial_rotary_attention

        if not (
            o2.Irreps.common_multiplicity(self.irreps_in)
            == o2.Irreps.common_multiplicity(self.irreps_out)
            == num_channel
        ):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")

        self.local_frame_in = o2.LocalFrame(self.irreps_in, self.lmax, self.mmax)
        self.local_frame_out = o2.LocalFrame(self.irreps_out, self.lmax, self.mmax)
        self.reshape_in = LayoutTransform(self.irreps_in)
        self.reshape_out = LayoutTransform(self.irreps_out)
        node_local_irreps_in = o2.Irreps(
            [(mul // num_channel, ir) for mul, ir in self.local_frame_in.local_irreps]
        )
        node_local_irreps_out = o2.Irreps(
            [(mul // num_channel, ir) for mul, ir in self.local_frame_out.local_irreps]
        )

        self.use_radial_rotary_attention = (
            use_radial_rotary_attention
            and self.local_frame_in.local_irreps.mmax > 0
        )
        if self.use_radial_rotary_attention and (
            self.num_head < 1 or self.num_channel % self.num_head != 0
        ):
            raise ValueError("num_head must divide num_channel.")

        self.local_irreps_in = o2.Irreps(
            tuple(node_local_irreps_in) * 2
        ).regroup()

        self.local_irreps_out = node_local_irreps_out
        self.use_asymmetric_contraction = (
            use_asymmetric_contraction
            and self.irreps_in.lmax > 0
        )

        self.node_feature_indices = {
            ir: index
            for index, (_, ir) in enumerate(node_local_irreps_in)
        }
        if self.use_asymmetric_contraction:
            self.nonlinearity = None
            contraction_irreps = o2.Irreps(
                [ir for _, ir in self.local_irreps_out]
            )
            self.asymmetric_contraction = o2.AsymmetricContraction(
                contraction_irreps,
                contraction_irreps,
                self.edge_ace_hidden,
                self.correlation,
                algorithm="edge",
                path_mode=self.ece_path_mode,
            )
            self.scalar_act = act_0e
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                (
                    contraction_irreps
                    * (self.correlation * self.edge_ace_hidden)
                    + o2.Irreps("0e")
                    * self.asymmetric_contraction.weight_numel
                ).regroup(),
                num_channel,
                1,
                path_mode="uv",
                bias=True,
            )
        else:
            self.asymmetric_contraction = None
            hidden_irreps = self.local_irreps_out.filter(
                keep=lambda mul_ir: self.local_irreps_in.count(mul_ir.ir) > 0
            )
            self.nonlinearity = o2.Gate(
                num_channel * hidden_irreps,
                act_0e=act_0e,
                act_0o=act_0o,
                act_lm=act_lm,
            )
            projection_irreps = o2.Irreps(
                [
                    (mul // num_channel, ir)
                    for mul, ir in self.nonlinearity.irreps_in
                ]
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                projection_irreps,
                num_channel,
                path_mode="uv",
                bias=True,
            )
        if self.asymmetric_contraction is not None:
            nonlinear_irreps_out = self.asymmetric_contraction.irreps_out
        else:
            nonlinear_irreps_out = o2.Irreps(
                [
                    (mul // num_channel, ir)
                    for mul, ir in self.nonlinearity.irreps_out
                ]
            )
        self.linear_down = o2.Linear(
            nonlinear_irreps_out,
            self.local_irreps_out,
            self.edge_ace_hidden
            if self.asymmetric_contraction is not None
            else num_channel,
            num_channel,
            path_mode="uv",
            bias=True,
        )
        self.weight_numel = self.linear_up.irreps_in.num_irreps * num_channel

        if self.use_radial_rotary_attention:
            self.attention = RadialRotaryComplexAttention(
                node_local_irreps_in,
                self.linear_down.irreps_out,
                num_channel,
                self.num_head,
                num_radial_basis,
            )
        else:
            self.attention = None

    def _apply_asymmetric_contraction(
        self,
        projected_features: tuple[torch.Tensor, ...],
        contraction_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.asymmetric_contraction is None:
            raise RuntimeError("O2 asymmetric contraction is not enabled.")
        if len(projected_features) != len(self.asymmetric_contraction.irreps_in):
            raise ValueError("Expected one projected tensor per contraction irrep.")
        contraction_inputs = [[] for _ in range(self.correlation)]
        for (mul, _), features in zip(
            self.asymmetric_contraction.irreps_in,
            projected_features,
        ):
            for correlation_index in range(self.correlation):
                start = (
                    correlation_index * mul * self.edge_ace_hidden
                )
                stop = start + mul * self.edge_ace_hidden
                contraction_inputs[correlation_index].append(
                    features[..., start:stop]
                )
        dense_inputs = [
            _to_contraction_layout(
                tuple(features),
                self.asymmetric_contraction.irreps_in,
                self.edge_ace_hidden,
            )
            for features in contraction_inputs
        ]
        contracted = self.asymmetric_contraction(
            dense_inputs,
            self.scalar_act(contraction_weights),
        )
        return _from_contraction_layout(
            contracted,
            self.asymmetric_contraction.irreps_out,
            self.edge_ace_hidden,
        )

    def _to_local(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        node_features = self.reshape_in(node_features)
        paired_features = self.local_frame_in.to_local(
            node_features[edge_index.T].movedim(1, 2).contiguous(),
            wigner,
        )
        source_features = tuple(features[:, :, 0] for features in paired_features)
        target_features = tuple(features[:, :, 1] for features in paired_features)
        return node_features, source_features, target_features

    def _convolution(
        self,
        source_features: tuple[torch.Tensor, ...],
        target_features: tuple[torch.Tensor, ...],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Union[torch.Tensor, None],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, ...]:
        edge_features = tuple(
            torch.cat(
                (
                    target_features[self.node_feature_indices[ir]],
                    source_features[self.node_feature_indices[ir]],
                ),
                dim=-1,
            )
            for _, ir in self.linear_up.irreps_in
        )

        weighted_features = []
        offset = 0
        for (mul, _), features in zip(
            self.linear_up.irreps_in,
            edge_features,
        ):
            width = mul * self.num_channel
            weight = conv_weights[:, offset : offset + width].unsqueeze(-2)
            weighted_features.append(features * weight)
            offset += width
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")

        projected_features = self.linear_up.forward_grouped(
            tuple(weighted_features)
        )
        if self.asymmetric_contraction is not None:
            projected_channels = self.correlation * self.edge_ace_hidden
            contraction_weights = projected_features[0][
                ..., projected_channels:
            ].reshape(
                conv_weights.size(0),
                self.asymmetric_contraction.weight_numel,
            )
            projected_features = tuple(
                features[..., :projected_channels]
                for features in projected_features
            )
            hidden_features = self._apply_asymmetric_contraction(
                projected_features,
                contraction_weights,
            )
        else:
            if self.nonlinearity is None:
                raise RuntimeError("O2 nonlinearity is not configured.")
            hidden_features = self.nonlinearity.forward_grouped(projected_features)
        message = self.linear_down.forward_grouped(hidden_features)
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
        message: tuple[torch.Tensor, ...],
        edge_index: torch.Tensor,
        wigner_inv: Union[torch.Tensor, None],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        message = self.local_frame_out.to_global(message, wigner_inv)
        if self.attention is None:
            message = message * edge_cutoff.unsqueeze(-1)
        return self.reshape_out.inverse(
            scatter_sum(
                message,
                edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: Union[torch.Tensor, None] = None,
        edge_cutoff: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        node_features, source_features, target_features = self._to_local(
            node_feats,
            edge_index,
            wigner,
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
    """O2 tensor product including magnetic solid harmonics followed by scatter."""

    ece_path_mode: str = "expand"  # [expand, sum]

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
        act_0o: Union[torch.nn.Module, None],
        act_lm: torch.nn.Module,
        correlation: int,
        num_head: int,
        num_radial_basis: int,
        use_asymmetric_contraction: bool,
        use_radial_rotary_attention: bool,
        edge_ace_hidden: Union[int, None] = None,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_irreps = o3.Irreps(magnetic_irreps)
        self.num_channel = num_channel
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        self.lmax = lmax
        self.mmax = min(
            max(self.irreps_in.lmax, self.magnetic_irreps.lmax),
            mmax,
        )
        self.correlation = correlation
        self.num_head = num_head
        self.use_asymmetric_contraction = use_asymmetric_contraction
        self.use_radial_rotary_attention = use_radial_rotary_attention

        if not (
            o2.Irreps.common_multiplicity(self.irreps_in)
            == o2.Irreps.common_multiplicity(self.irreps_out)
            == num_channel
        ):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")

        self.local_frame_in = o2.LocalFrame(self.irreps_in, self.lmax, self.mmax)
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            self.lmax,
            self.mmax,
        )
        self.magnetic_frame = o2.LocalFrame(
            self.magnetic_irreps,
            self.lmax,
            self.mmax,
        )
        self.reshape_in = LayoutTransform(self.irreps_in)
        self.reshape_out = LayoutTransform(self.irreps_out)
        self.reshape_magnetic = LayoutTransform(self.magnetic_irreps)

        node_local_irreps_in = o2.Irreps(
            [(mul // num_channel, ir) for mul, ir in self.local_frame_in.local_irreps]
        )
        node_local_irreps_out = o2.Irreps(
            [
                (mul // num_channel, ir)
                for mul, ir in self.local_frame_out.local_irreps
            ]
        )
        self.local_irreps_in = o2.Irreps(
            tuple(node_local_irreps_in) * 2
            + tuple(self.magnetic_frame.local_irreps) * 2
        ).regroup()
        self.local_irreps_out = node_local_irreps_out

        self.use_radial_rotary_attention = (
            use_radial_rotary_attention
            and self.local_frame_in.local_irreps.mmax > 0
        )
        if self.use_radial_rotary_attention and (
            self.num_head < 1 or self.num_channel % self.num_head != 0
        ):
            raise ValueError("num_head must divide num_channel.")

        self.use_asymmetric_contraction = (
            use_asymmetric_contraction
            and self.irreps_in.lmax > 0
        )
        self.node_feature_indices = {
            ir: index
            for index, (_, ir) in enumerate(node_local_irreps_in)
        }
        self.magnetic_feature_indices = {
            ir: index
            for index, (_, ir) in enumerate(self.magnetic_frame.local_irreps)
        }

        if self.use_asymmetric_contraction:
            self.nonlinearity = None
            contraction_irreps = o2.Irreps(
                [ir for _, ir in self.local_irreps_out]
            )
            self.asymmetric_contraction = o2.AsymmetricContraction(
                contraction_irreps,
                contraction_irreps,
                self.edge_ace_hidden,
                self.correlation,
                algorithm="edge",
                path_mode=self.ece_path_mode,
            )
            self.scalar_act = act_0e
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                (
                    contraction_irreps
                    * (self.correlation * self.edge_ace_hidden)
                    + o2.Irreps("0e")
                    * self.asymmetric_contraction.weight_numel
                ).regroup(),
                num_channel,
                1,
                path_mode="uv",
                bias=True,
            )
        else:
            self.asymmetric_contraction = None
            hidden_irreps = self.local_irreps_out.filter(
                keep=lambda mul_ir: self.local_irreps_in.count(mul_ir.ir) > 0
            )
            self.nonlinearity = o2.Gate(
                num_channel * hidden_irreps,
                act_0e=act_0e,
                act_0o=act_0o,
                act_lm=act_lm,
            )
            projection_irreps = o2.Irreps(
                [
                    (mul // num_channel, ir)
                    for mul, ir in self.nonlinearity.irreps_in
                ]
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                projection_irreps,
                num_channel,
                path_mode="uv",
                bias=True,
            )

        if self.asymmetric_contraction is not None:
            nonlinear_irreps_out = self.asymmetric_contraction.irreps_out
        else:
            nonlinear_irreps_out = o2.Irreps(
                [
                    (mul // num_channel, ir)
                    for mul, ir in self.nonlinearity.irreps_out
                ]
            )
        self.linear_down = o2.Linear(
            nonlinear_irreps_out,
            self.local_irreps_out,
            self.edge_ace_hidden
            if self.asymmetric_contraction is not None
            else num_channel,
            num_channel,
            path_mode="uv",
            bias=True,
        )
        self.weight_numel = self.linear_up.irreps_in.num_irreps * num_channel

        if self.use_radial_rotary_attention:
            self.attention = RadialRotaryComplexAttention(
                node_local_irreps_in,
                self.linear_down.irreps_out,
                num_channel,
                self.num_head,
                num_radial_basis,
            )
        else:
            self.attention = None

    def _apply_asymmetric_contraction(
        self,
        projected_features: tuple[torch.Tensor, ...],
        contraction_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.asymmetric_contraction is None:
            raise RuntimeError("O2 asymmetric contraction is not enabled.")
        if len(projected_features) != len(self.asymmetric_contraction.irreps_in):
            raise ValueError("Expected one projected tensor per contraction irrep.")
        contraction_inputs = [[] for _ in range(self.correlation)]
        for (mul, _), features in zip(
            self.asymmetric_contraction.irreps_in,
            projected_features,
        ):
            for correlation_index in range(self.correlation):
                start = correlation_index * mul * self.edge_ace_hidden
                stop = start + mul * self.edge_ace_hidden
                contraction_inputs[correlation_index].append(
                    features[..., start:stop]
                )
        dense_inputs = [
            _to_contraction_layout(
                tuple(features),
                self.asymmetric_contraction.irreps_in,
                self.edge_ace_hidden,
            )
            for features in contraction_inputs
        ]
        contracted = self.asymmetric_contraction(
            dense_inputs,
            self.scalar_act(contraction_weights),
        )
        return _from_contraction_layout(
            contracted,
            self.asymmetric_contraction.irreps_out,
            self.edge_ace_hidden,
        )

    def _to_local(
        self,
        node_features: torch.Tensor,
        magnetic_node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        node_features = self.reshape_in(node_features)
        paired_node_features = self.local_frame_in.to_local(
            node_features[edge_index.T].movedim(1, 2).contiguous(),
            wigner,
        )
        magnetic_node_attrs = self.reshape_magnetic(magnetic_node_attrs)
        paired_magnetic_features = self.magnetic_frame.to_local(
            magnetic_node_attrs[edge_index.T].movedim(1, 2).contiguous(),
            wigner,
        )
        source_features = tuple(
            features[:, :, 0] for features in paired_node_features
        )
        target_features = tuple(
            features[:, :, 1] for features in paired_node_features
        )
        source_magnetic_features = tuple(
            features[:, :, 0].repeat_interleave(self.num_channel, dim=-1)
            for features in paired_magnetic_features
        )
        target_magnetic_features = tuple(
            features[:, :, 1].repeat_interleave(self.num_channel, dim=-1)
            for features in paired_magnetic_features
        )
        return (
            node_features,
            source_features,
            target_features,
            source_magnetic_features,
            target_magnetic_features,
        )

    def _convolution(
        self,
        source_features: tuple[torch.Tensor, ...],
        target_features: tuple[torch.Tensor, ...],
        source_magnetic_features: tuple[torch.Tensor, ...],
        target_magnetic_features: tuple[torch.Tensor, ...],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Union[torch.Tensor, None],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, ...]:
        edge_features = []
        for _, ir in self.linear_up.irreps_in:
            parts = []
            node_index = self.node_feature_indices.get(ir)
            if node_index is not None:
                parts.extend(
                    (
                        target_features[node_index],
                        source_features[node_index],
                    )
                )
            magnetic_index = self.magnetic_feature_indices.get(ir)
            if magnetic_index is not None:
                parts.extend(
                    (
                        target_magnetic_features[magnetic_index],
                        source_magnetic_features[magnetic_index],
                    )
                )
            edge_features.append(torch.cat(parts, dim=-1))

        weighted_features = []
        offset = 0
        for (mul, _), features in zip(
            self.linear_up.irreps_in,
            edge_features,
        ):
            width = mul * self.num_channel
            weight = conv_weights[:, offset : offset + width].unsqueeze(-2)
            weighted_features.append(features * weight)
            offset += width
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")

        projected_features = self.linear_up.forward_grouped(
            tuple(weighted_features)
        )
        if self.asymmetric_contraction is not None:
            projected_channels = self.correlation * self.edge_ace_hidden
            contraction_weights = projected_features[0][
                ..., projected_channels:
            ].reshape(
                conv_weights.size(0),
                self.asymmetric_contraction.weight_numel,
            )
            projected_features = tuple(
                features[..., :projected_channels]
                for features in projected_features
            )
            hidden_features = self._apply_asymmetric_contraction(
                projected_features,
                contraction_weights,
            )
        else:
            if self.nonlinearity is None:
                raise RuntimeError("O2 nonlinearity is not configured.")
            hidden_features = self.nonlinearity.forward_grouped(projected_features)
        message = self.linear_down.forward_grouped(hidden_features)
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
        message: tuple[torch.Tensor, ...],
        edge_index: torch.Tensor,
        wigner_inv: Union[torch.Tensor, None],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        message = self.local_frame_out.to_global(message, wigner_inv)
        if self.attention is None:
            message = message * edge_cutoff.unsqueeze(-1)
        return self.reshape_out.inverse(
            scatter_sum(
                message,
                edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        magnetic_node_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: Union[torch.Tensor, None] = None,
        edge_cutoff: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        (
            node_features,
            source_features,
            target_features,
            source_magnetic_features,
            target_magnetic_features,
        ) = self._to_local(
            node_feats,
            magnetic_node_attrs,
            edge_index,
            wigner,
        )
        message = self._convolution(
            source_features,
            target_features,
            source_magnetic_features,
            target_magnetic_features,
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
