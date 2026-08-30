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
        self.q_proj = o2.Linear(
            self.irreps,
            self.irreps,
            channels,
            channels,
            path_mode="uv",
            bias=False,
        )
        self.k_proj = o2.Linear(
            self.irreps,
            self.irreps,
            channels,
            channels,
            path_mode="uv",
            bias=False,
        )
        self.radial_proj = torchLinear(
            num_radial_basis,
            2 * num_head,
        )
        torch.nn.init.zeros_(self.radial_proj.weight)
        torch.nn.init.zeros_(self.radial_proj.bias)
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
        query_blocks = self.q_proj.forward_grouped(target_blocks)
        key_blocks = self.k_proj.forward_grouped(source_blocks)
        return query_blocks, key_blocks

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
    """Local O2 Convolution.

    Source and target node features are concatenated in the local
    representation. Optional extra node attributes use the same source-target
    construction. Edge weights first apply a diagonal channel-wise radial map.
    Two internal ``uv`` linears use either a gate or an asymmetric contraction
    between them. These nonlinear paths are mutually exclusive. The
    asymmetric path keeps one copy of each intermediate irrep and stores its
    latent width in ``edge_ace_hidden`` channels; the gate path retains
    ``num_channel`` throughout. The first O2 linear also generates auxiliary
    ``0e`` gates for the positive-order representations. Those gates use the
    tensor activation. The optional
    attention uses real O2-invariant query-key products and a zero-initialized
    radial scale-and-shift projection whose sigmoid scale starts at one; it
    never applies a complex phase.
    Radial weights never contain the cutoff. Without attention, the cutoff is
    applied to the global O3 edge message; with attention, it is applied before
    the inverse local-to-global transformation.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
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
        edge_ace_hidden: Union[int, None] = None,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.extra_node_attrs_irreps = (
            None
            if extra_node_attrs_irreps is None
            else o3.Irreps(extra_node_attrs_irreps)
        )
        self.num_channel = num_channel
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        self.lmax = lmax
        self.mmax = mmax
        self.correlation = correlation
        self.num_head = num_head
        self.use_asymmetric_contraction = use_asymmetric_contraction
        self.use_radial_rotary_attention = use_radial_rotary_attention
        if (
            o2.Irreps.common_multiplicity(self.irreps_in)
            != num_channel
        ):
            raise ValueError("irreps_in multiplicity must equal num_channel.")
        if (
            o2.Irreps.common_multiplicity(self.irreps_out)
            != num_channel
        ):
            raise ValueError("irreps_out multiplicity must equal num_channel.")
        if self.correlation < 1:
            raise ValueError("correlation must be positive.")

        self.effective_mmax = min(
            max(
                self.irreps_in.lmax,
                self.extra_node_attrs_irreps.lmax
                if self.extra_node_attrs_irreps is not None
                else 0,
            ),
            mmax,
        )

        self.local_frame_in = o2.LocalFrame(
            self.irreps_in,
            lmax,
            self.effective_mmax,
        )
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            lmax,
            self.effective_mmax,
        )
        self.use_radial_rotary_attention = (
            use_radial_rotary_attention
            and self.local_frame_in.local_irreps.mmax > 0
        )
        if self.use_radial_rotary_attention and (
            self.num_head < 1 or self.num_channel % self.num_head != 0
        ):
            raise ValueError("num_head must divide num_channel.")

        if self.extra_node_attrs_irreps is not None:
            self.local_frame_extra = o2.LocalFrame(
                self.extra_node_attrs_irreps,
                lmax,
                self.effective_mmax,
            )
        self.local_irreps_in = o2.Irreps(
            tuple(self.local_frame_in.local_irreps) * 2
            + (
                tuple(self.local_frame_extra.local_irreps) * 2
                if self.extra_node_attrs_irreps is not None
                else ()
            )
        ).regroup()

        self.local_irreps_out = self.local_frame_out.local_irreps
        self.use_asymmetric_contraction = (
            use_asymmetric_contraction
            and self.irreps_in.lmax > 0
        )

        self.node_block_indices = {
            irrep: index
            for index, (_, irrep) in enumerate(
                self.local_frame_in.local_irreps
            )
        }
        self.extra_node_attrs_block_indices = (
            {
                irrep: index
                for index, (_, irrep) in enumerate(
                    self.local_frame_extra.local_irreps
                )
            }
            if self.extra_node_attrs_irreps is not None
            else {}
        )

        if self.use_asymmetric_contraction:
            self.nonlinearity = None
            contraction_irreps = o2.Irreps(
                [irrep for _, irrep in self.local_irreps_out]
            )
            self.asymmetric_contraction = o2.AsymmetricContraction(
                contraction_irreps,
                contraction_irreps,
                self.edge_ace_hidden,
                self.correlation,
                algorithm="edge",
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
                bias=False,
            )
        else:
            self.asymmetric_contraction = None
            hidden_irreps = o2.Irreps(
                [
                    (multiplicity, irrep)
                    for multiplicity, irrep in self.local_irreps_out
                    if self.local_irreps_in.count(irrep) > 0
                ]
            )
            self.nonlinearity = o2.Gate(
                num_channel * hidden_irreps,
                act_0e=act_0e,
                act_0o=act_0o,
                act_lm=act_lm,
            )
            projection_irreps = o2.Irreps(
                [
                    (multiplicity // num_channel, irrep)
                    for multiplicity, irrep in self.nonlinearity.irreps_in
                ]
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                projection_irreps,
                num_channel,
                path_mode="uv",
                bias=False,
            )
        if self.asymmetric_contraction is not None:
            nonlinear_irreps_out = self.asymmetric_contraction.irreps_out
        else:
            nonlinear_irreps_out = o2.Irreps(
                [
                    (multiplicity // num_channel, irrep)
                    for multiplicity, irrep in self.nonlinearity.irreps_out
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
            bias=False,
        )
        self.weight_numel = self.linear_up.irreps_in.num_irreps * num_channel

        if self.use_radial_rotary_attention:
            self.attention = RadialRotaryComplexAttention(
                self.local_frame_in.local_irreps,
                self.linear_down.irreps_out,
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
        contraction_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.asymmetric_contraction is None:
            raise RuntimeError("O2 asymmetric contraction is not enabled.")
        if len(projected_blocks) != len(self.asymmetric_contraction.irreps_in):
            raise ValueError("Expected one projected block per contraction irrep.")
        contraction_inputs = [[] for _ in range(self.correlation)]
        for (multiplicity, _), block in zip(
            self.asymmetric_contraction.irreps_in,
            projected_blocks,
        ):
            for correlation_index in range(self.correlation):
                start = (
                    correlation_index * multiplicity * self.edge_ace_hidden
                )
                stop = start + multiplicity * self.edge_ace_hidden
                contraction_inputs[correlation_index].append(
                    block[..., start:stop]
                )
        dense_inputs = [
            self._grouped_to_dense(
                tuple(blocks),
                self.asymmetric_contraction.irreps_in,
                self.edge_ace_hidden,
            )
            for blocks in contraction_inputs
        ]
        contracted = self.asymmetric_contraction(
            dense_inputs,
            self.scalar_act(contraction_weights),
        )
        return self._dense_to_grouped(
            contracted,
            self.asymmetric_contraction.irreps_out,
            self.edge_ace_hidden,
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
        source_node_blocks = self.local_frame_in.to_local(
            node_feats[source],
            wigner,
        )
        target_node_blocks = self.local_frame_in.to_local(
            node_feats[target],
            wigner,
        )
        source_extra_blocks = None
        target_extra_blocks = None
        if self.extra_node_attrs_irreps is not None:
            if extra_node_attrs is None:
                raise ValueError("O2 convolution requires extra_node_attrs.")
            source_extra_blocks = self.local_frame_extra.to_local(
                extra_node_attrs[source],
                wigner,
            )
            target_extra_blocks = self.local_frame_extra.to_local(
                extra_node_attrs[target],
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
        for _, irrep in self.linear_up.irreps_in:
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
            self.linear_up.irreps_in,
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
        if self.asymmetric_contraction is not None:
            projected_channels = self.correlation * self.edge_ace_hidden
            contraction_weights = projected_blocks[0][
                ..., projected_channels:
            ].reshape(
                radial_weights.size(0),
                self.asymmetric_contraction.weight_numel,
            )
            projected_blocks = tuple(
                block[..., :projected_channels]
                for block in projected_blocks
            )
            hidden_blocks = self._apply_asymmetric_contraction(
                projected_blocks,
                contraction_weights,
            )
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
        messages = self.local_frame_out.to_global(output_blocks, wigner_inv)
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
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_irreps: o3.Irreps,
        **kwargs,
    ) -> None:
        magnetic_irreps = o3.Irreps(magnetic_irreps)
        super().__init__(
            irreps_in,
            irreps_out,
            extra_node_attrs_irreps=magnetic_irreps,
            **kwargs,
        )
        self.magnetic_irreps = magnetic_irreps

    @property
    def magnetic_frame(self) -> o2.LocalFrame:
        return self.local_frame_extra

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
