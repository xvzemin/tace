################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Complete local O(2) interactions for the e3nn model."""

import math
from typing import Union

import torch
from e3nn import o3

from eqx import o2

from tace.utils.torch_scatter import scatter_sum

from ..linear import torchLinear
from ..softmax import GraphSoftmax


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
        self.qk_proj = o2.Linear(
            self.irreps,
            self.irreps,
            channels,
            2 * channels,
            path_mode="uv",
            bias=False,
        )
        self.radial_scale_shift = torchLinear(
            num_radial_basis,
            2 * num_head,
        )
        torch.nn.init.zeros_(self.radial_scale_shift.weight)
        torch.nn.init.zeros_(self.radial_scale_shift.bias)
        components_per_head = self.irreps.dim * channels // num_head
        self.scale = 1.0 / math.sqrt(components_per_head)
        self.graph_softmax = GraphSoftmax()

    def _project_query_key(
        self,
        source_blocks: tuple[torch.Tensor, ...],
        target_blocks: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if len(source_blocks) != len(self.irreps) or len(target_blocks) != len(
            self.irreps
        ):
            raise ValueError("Expected one source and target block per O2 group.")
        num_edges = source_blocks[0].size(0)
        combined_blocks = tuple(
            torch.cat((target, source), dim=0)
            for source, target in zip(source_blocks, target_blocks)
        )
        projected_blocks = self.qk_proj.forward_grouped(combined_blocks)
        query_blocks = []
        key_blocks = []
        for projected, (multiplicity, irrep) in zip(projected_blocks, self.irreps):
            target_projected, source_projected = projected.split(num_edges, dim=0)
            leading_shape = target_projected.shape[:-2]
            target_projected = target_projected.reshape(
                *leading_shape,
                irrep.dim,
                multiplicity,
                2 * self.channels,
            )
            source_projected = source_projected.reshape(
                *leading_shape,
                irrep.dim,
                multiplicity,
                2 * self.channels,
            )
            query = target_projected[..., : self.channels]
            key = source_projected[..., self.channels :]
            query_blocks.append(
                query.reshape(
                    *leading_shape,
                    irrep.dim,
                    multiplicity * self.channels,
                )
            )
            key_blocks.append(
                key.reshape(
                    *leading_shape,
                    irrep.dim,
                    multiplicity * self.channels,
                )
            )
        return tuple(query_blocks), tuple(key_blocks)

    def forward(
        self,
        message_blocks: tuple[torch.Tensor, ...],
        source_blocks: tuple[torch.Tensor, ...],
        target_blocks: tuple[torch.Tensor, ...],
        edge_radial_basis: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, ...]:
        if len(message_blocks) != len(self.message_irreps):
            raise ValueError("Expected one message block per output O2 group.")
        query_blocks, key_blocks = self._project_query_key(
            source_blocks,
            target_blocks,
        )
        score = edge_radial_basis.new_zeros(
            edge_radial_basis.size(0),
            self.num_head,
        )
        for query, key, (multiplicity, irrep) in zip(
            query_blocks,
            key_blocks,
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
        radial_scale, radial_shift = self.radial_scale_shift(
            edge_radial_basis
        ).chunk(2, dim=-1)
        score = (
            score * self.scale * torch.sigmoid(radial_scale) + radial_shift
        )

        attention = self.graph_softmax(
            score,
            edge_index[1],
            num_nodes=num_nodes,
            exp_rescale=edge_cutoff,
        )
        attention = attention * edge_cutoff

        outputs = []
        for block, (multiplicity, irrep) in zip(
            message_blocks,
            self.message_irreps,
        ):
            shape = (
                block.size(0),
                irrep.dim,
                multiplicity,
                self.num_head,
                self.channels_per_head,
            )
            output = block.reshape(shape) * attention[:, None, None, :, None]
            outputs.append(output.reshape_as(block))
        return tuple(outputs)


class O2ScatterLinear(torch.nn.Module):
    """Edge-aligned O2 gated convolution.

    Source and target node features are concatenated in the local
    representation. Optional extra node attributes use the same source-target
    construction. Edge weights first apply a diagonal channel-wise radial map.
    Two internal ``uv`` linears use either a gate or an asymmetric contraction
    between them. These nonlinear paths are mutually exclusive. The first O2
    linear also generates auxiliary ``0e`` gates for the positive-order
    representations. Those gates use the tensor activation. The optional
    attention uses real O2-invariant query-key products and a zero-initialized
    radial scale-and-shift projection; it never applies a complex phase.
    Radial weights never contain the cutoff. Without attention, the cutoff is
    applied to the global O3 edge message; with attention, it is applied before
    the inverse local-to-global transformation.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        extra_node_attrs_irreps: Union[o3.Irreps, None] = None,
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
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_out = o3.Irreps(irreps_out)
        self.extra_node_attrs_irreps = (
            None
            if extra_node_attrs_irreps is None
            else o3.Irreps(extra_node_attrs_irreps)
        )
        self.num_channel = num_channel
        self.lmax = lmax
        self.mmax = mmax
        self.correlation = correlation
        self.num_head = num_head
        self.use_asymmetric_contraction = use_asymmetric_contraction
        self.use_radial_rotary_attention = use_radial_rotary_attention
        if (
            o2.Irreps.common_multiplicity(self.irreps_node)
            != num_channel
        ):
            raise ValueError("irreps_node multiplicity must equal num_channel.")
        if (
            o2.Irreps.common_multiplicity(self.irreps_out)
            != num_channel
        ):
            raise ValueError("irreps_out multiplicity must equal num_channel.")
        if self.correlation < 1:
            raise ValueError("correlation must be positive.")
        if self.use_radial_rotary_attention and (
            self.num_head < 1 or self.num_channel % self.num_head != 0
        ):
            raise ValueError("num_head must divide num_channel.")

        input_lmax = self.irreps_node.lmax
        if self.extra_node_attrs_irreps is not None:
            input_lmax = max(input_lmax, self.extra_node_attrs_irreps.lmax)
        self.active_mmax = min(input_lmax, mmax)

        self.reshape_in = o2.O3O2Layout(
            self.irreps_node,
            lmax,
            self.active_mmax,
        )
        self.reshape_out = o2.O3O2Layout(
            self.irreps_out,
            lmax,
            self.active_mmax,
        )
        input_groups = (
            tuple(self.reshape_in.local_irreps)
            + tuple(self.reshape_in.local_irreps)
        )
        if self.extra_node_attrs_irreps is not None:
            self.extra_node_attrs_layout = o2.O3O2Layout(
                self.extra_node_attrs_irreps,
                lmax,
                self.active_mmax,
            )
            input_groups += (
                tuple(self.extra_node_attrs_layout.local_irreps)
                + tuple(self.extra_node_attrs_layout.local_irreps)
            )
        self.irreps_in_local = o2.Irreps(input_groups).regroup()
        self.irreps_out_local = self.reshape_out.local_irreps

        # TODO start from here
        self.input_block_irreps = tuple(irrep for _, irrep in self.irreps_in_local)
        self.node_block_indices = {
            irrep: index
            for index, (_, irrep) in enumerate(self.reshape_in.local_irreps)
        }
        self.extra_node_attrs_block_indices = (
            {
                irrep: index
                for index, (_, irrep) in enumerate(
                    self.extra_node_attrs_layout.local_irreps
                )
            }
            if self.extra_node_attrs_irreps is not None
            else {}
        )

        if self.use_asymmetric_contraction:
            self.nonlinearity = None
            self.asymmetric_contraction = o2.O2AsymmetricContraction(
                self.irreps_out_local,
                self.irreps_out_local,
                num_channel,
                self.correlation,
                algorithm="edge",
            )
            self.contraction_weight_numel = self.asymmetric_contraction.weight_numel
            self.scalar_act = act_0e
            projection_groups = [
                (multiplicity * self.correlation, irrep)
                for multiplicity, irrep in self.irreps_out_local
            ]
            projection_groups.append(
                (self.asymmetric_contraction.num_paths, o2.Irrep("0e"))
            )
            self.projection_irreps = o2.Irreps(projection_groups).regroup()
            self.linear_up = o2.Linear(
                self.irreps_in_local,
                self.projection_irreps,
                num_channel,
                path_mode="uv",
                bias=False,
            )
        else:
            self.asymmetric_contraction = None
            self.contraction_weight_numel = 0
            self.irreps_hidden_local = o2.Irreps(
                [
                    (multiplicity, irrep)
                    for multiplicity, irrep in self.irreps_out_local
                    if self.irreps_in_local.count(irrep) > 0
                ]
            )
            self.nonlinearity = o2.O2Gate(
                num_channel * self.irreps_hidden_local,
                act_0e=act_0e,
                act_0o=act_0o,
                act_lm=act_lm,
            )
            self.projection_irreps = o2.Irreps(
                [
                    (multiplicity // num_channel, irrep)
                    for multiplicity, irrep in self.nonlinearity.irreps_in
                ]
            )
            self.linear_up = o2.Linear(
                self.irreps_in_local,
                self.projection_irreps,
                num_channel,
                path_mode="uv",
                bias=False,
            )
        self.linear_down = o2.Linear(
            (
                self.irreps_out_local
                if self.use_asymmetric_contraction
                else self.irreps_hidden_local
            ),
            self.irreps_out_local,
            num_channel,
            path_mode="uv",
            bias=False,
        )
        self.weight_numel = self.irreps_in_local.num_irreps * num_channel

        if self.use_radial_rotary_attention:
            self.attention = RadialRotaryComplexAttention(
                self.reshape_in.local_irreps,
                self.irreps_out_local,
                num_channel,
                self.num_head,
                num_radial_basis,
            )
        else:
            self.attention = None

    @staticmethod
    def _grouped_to_dense(
        blocks: tuple[torch.Tensor, ...],
        irreps: o2.Irreps,
        channels: int,
    ) -> torch.Tensor:
        if len(blocks) != len(irreps):
            raise ValueError("Expected one block per O(2) irrep group.")
        outputs = []
        for block, (multiplicity, irrep) in zip(blocks, irreps):
            leading_shape = block.shape[:-2]
            output = block.reshape(
                *leading_shape,
                irrep.dim,
                multiplicity,
                channels,
            )
            output = output.permute(
                *range(len(leading_shape)),
                len(leading_shape) + 1,
                len(leading_shape),
                len(leading_shape) + 2,
            )
            outputs.append(
                output.reshape(*leading_shape, multiplicity * irrep.dim, channels)
            )
        return torch.cat(outputs, dim=-2)

    @staticmethod
    def _dense_to_grouped(
        input: torch.Tensor,
        irreps: o2.Irreps,
        channels: int,
    ) -> tuple[torch.Tensor, ...]:
        leading_shape = input.shape[:-2]
        outputs = []
        offset = 0
        for multiplicity, irrep in irreps:
            width = multiplicity * irrep.dim
            block = input[..., offset : offset + width, :].reshape(
                *leading_shape,
                multiplicity,
                irrep.dim,
                channels,
            )
            block = block.permute(
                *range(len(leading_shape)),
                len(leading_shape) + 1,
                len(leading_shape),
                len(leading_shape) + 2,
            )
            outputs.append(
                block.reshape(*leading_shape, irrep.dim, multiplicity * channels)
            )
            offset += width
        if offset != input.size(-2):
            raise ValueError("Invalid dense O(2) representation size.")
        return tuple(outputs)

    def _apply_asymmetric_contraction(
        self,
        projected_blocks: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if self.asymmetric_contraction is None:
            raise RuntimeError("O2 asymmetric contraction is not enabled.")

        blocks_by_irrep = {
            irrep: block
            for (_, irrep), block in zip(self.projection_irreps, projected_blocks)
        }
        contraction_inputs = [[] for _ in range(self.correlation)]
        contraction_weights = None
        for multiplicity, irrep in self.irreps_out_local:
            block = blocks_by_irrep[irrep]
            feature_width = multiplicity * self.correlation * self.num_channel
            features = block[..., :feature_width]
            for correlation_index in range(self.correlation):
                start = correlation_index * multiplicity * self.num_channel
                stop = start + multiplicity * self.num_channel
                contraction_inputs[correlation_index].append(
                    features[..., start:stop]
                )
            if irrep == o2.Irrep("0e"):
                contraction_weights = block[..., feature_width:].reshape(
                    *block.shape[:-2],
                    self.contraction_weight_numel,
                )

        if contraction_weights is None:
            raise RuntimeError(
                "O2 contraction projection did not produce 0e weights."
            )
        dense_inputs = [
            self._grouped_to_dense(
                tuple(blocks),
                self.irreps_out_local,
                self.num_channel,
            )
            for blocks in contraction_inputs
        ]
        contracted = self.asymmetric_contraction(
            dense_inputs,
            self.scalar_act(contraction_weights),
        )
        return self._dense_to_grouped(
            contracted,
            self.irreps_out_local,
            self.num_channel,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        radial_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        extra_node_attrs: Union[torch.Tensor, None] = None,
        edge_radial_basis: Union[torch.Tensor, None] = None,
        edge_cutoff: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        if wigner is None or wigner_inv is None:
            raise ValueError("O2 convolution requires edge Wigner matrices.")

        source, target = edge_index
        source_node_blocks = self.reshape_in(node_feats[source], wigner)
        target_node_blocks = self.reshape_in(node_feats[target], wigner)
        source_extra_blocks = None
        target_extra_blocks = None
        if self.extra_node_attrs_irreps is not None:
            if extra_node_attrs is None:
                raise ValueError("O2 convolution requires extra_node_attrs.")
            source_extra_blocks = self.extra_node_attrs_layout(
                extra_node_attrs[source].unsqueeze(-1),
                wigner,
            )
            target_extra_blocks = self.extra_node_attrs_layout(
                extra_node_attrs[target].unsqueeze(-1),
                wigner,
            )
            source_extra_blocks = tuple(
                block.reshape(*block.shape, 1)
                .expand(*block.shape, self.num_channel)
                .reshape(*block.shape[:-1], block.size(-1) * self.num_channel)
                for block in source_extra_blocks
            )
            target_extra_blocks = tuple(
                block.reshape(*block.shape, 1)
                .expand(*block.shape, self.num_channel)
                .reshape(*block.shape[:-1], block.size(-1) * self.num_channel)
                for block in target_extra_blocks
            )

        input_blocks = []
        for irrep in self.input_block_irreps:
            parts = []
            node_index = self.node_block_indices.get(irrep)
            if node_index is not None:
                parts.append(target_node_blocks[node_index])
                parts.append(source_node_blocks[node_index])
            extra_index = self.extra_node_attrs_block_indices.get(irrep)
            if extra_index is not None:
                parts.append(target_extra_blocks[extra_index])
                parts.append(source_extra_blocks[extra_index])
            input_blocks.append(torch.cat(parts, dim=-1))
        input_blocks = tuple(input_blocks)

        radial_weights = radial_weights.reshape(radial_weights.size(0), -1)
        weighted_blocks = []
        offset = 0
        for (multiplicity, _), input_block in zip(
            self.irreps_in_local,
            input_blocks,
        ):
            width = multiplicity * self.num_channel
            weight = radial_weights[:, offset : offset + width].unsqueeze(-2)
            weighted_blocks.append(input_block * weight)
            offset += width
        if offset != radial_weights.size(-1):
            raise ValueError("Invalid O2 radial weight size.")

        weighted_blocks = tuple(weighted_blocks)
        projected_blocks = self.linear_up.forward_grouped(weighted_blocks)
        if self.use_asymmetric_contraction:
            hidden_blocks = self._apply_asymmetric_contraction(projected_blocks)
        else:
            if self.nonlinearity is None:
                raise RuntimeError("O2 nonlinearity is not configured.")
            hidden_blocks = self.nonlinearity.forward_grouped(projected_blocks)
        output_blocks = self.linear_down.forward_grouped(hidden_blocks)
        if self.attention is not None:
            if edge_radial_basis is None:
                raise ValueError(
                    "O2 radial rotary attention requires edge_radial_basis."
                )
            output_blocks = self.attention(
                output_blocks,
                source_node_blocks,
                target_node_blocks,
                edge_radial_basis,
                edge_index,
                edge_cutoff,
                node_feats.size(0),
            )
        messages = self.reshape_out.inverse(output_blocks, wigner_inv)
        if self.attention is None:
            messages = messages * edge_cutoff
        return scatter_sum(
            messages,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )


class O2MagneticScatterLinear(O2ScatterLinear):
    """O2 scatter convolution with magnetic solid harmonics as extra attrs."""

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_irreps: o3.Irreps,
        **kwargs,
    ) -> None:
        magnetic_irreps = o3.Irreps(magnetic_irreps)
        super().__init__(
            irreps_node,
            irreps_out,
            extra_node_attrs_irreps=magnetic_irreps,
            **kwargs,
        )
        self.magnetic_irreps = magnetic_irreps

    @property
    def magnetic_layout(self) -> o2.O3O2Layout:
        return self.extra_node_attrs_layout

    def forward(
        self,
        node_feats: torch.Tensor,
        magnetic_node_attrs: torch.Tensor,
        radial_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: Union[torch.Tensor, None] = None,
        edge_cutoff: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        return super().forward(
            node_feats,
            radial_weights,
            edge_index,
            wigner,
            wigner_inv,
            magnetic_node_attrs,
            edge_radial_basis,
            edge_cutoff,
        )
