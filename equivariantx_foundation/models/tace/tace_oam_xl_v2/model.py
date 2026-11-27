"""Whole-model scheduling for OAM CGTP and local O(2) interactions.

Conversion preserves parameter objects and checkpoint names. Geometry-only
networks are evaluated together before the sequential node stack. The compiled
training boundary includes E/F/S, the caller's loss, and parameter derivatives.
No learned quantity is cached across optimizer updates.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from equivariantx.ops.common.channel_product import channel_product
from equivariantx.ops.common.weights import ConvolutionWeights
from equivariantx.utils.layout import permute

from ..matching import require_native_messages
from .angular import aligned_convolution, packed_frame
from .gated import gated_product, gated_product_metadata
from .layout import channel_contiguous_metadata, ir_mul_order
from .packed_product import (
    angular_metadata,
    composed_linear,
    packed_contract,
    packed_linear,
    packed_metadata,
)


@dataclass(frozen=True)
class Architecture:
    """Static contract used to select the whole-model execution schedule."""

    layers: int
    channels: int
    elements: int
    parameters: int
    parameter_names: tuple[str, ...]


def transplant(kind, source):
    """Replace execution while preserving the copied module's registration."""
    result = kind.__new__(kind)
    result.__dict__ = source.__dict__.copy()
    return result


class LinearMap(torch.nn.Module):
    """Apply each irrep matrix directly, without concatenating all weights."""

    def scalar_matrix(self):
        """Return the normalized matrix of a single scalar coupling."""
        (instruction,) = self.linear.instructions
        weight = (
            self.weight[0]
            if isinstance(self.weight, torch.nn.ParameterList)
            else self.weight
        )
        return weight.reshape(instruction.path_shape) * instruction.path_weight

    def forward(self, inputs, weight=None):
        if weight is not None:
            raise ValueError("TACE-OAM-XL-v2 owns its linear weights")
        if getattr(self, "native_linear", False):
            from .linear import irrep_linear

            return irrep_linear(self, inputs)
        matrices = self.weight
        if not isinstance(matrices, torch.nn.ParameterList):
            matrices = matrices.split([a * b for a, b in self._path_shapes])
        pieces = [[] for _ in self.irreps_out]
        input_slices = self.irreps_in.slices()
        input_blocks = None
        if os.environ.get("EQX_XL_LAYOUT_ADJOINTS", "0") == "1":
            from .layout_adjoint import ColumnSlices

            input_blocks = ColumnSlices.apply(
                inputs, tuple(c * ir.dim for c, ir in self.irreps_in)
            )
        for matrix, instruction in zip(matrices, self.linear.instructions):
            ci, ir = self.irreps_in[instruction.i_in]
            co, _ = self.irreps_out[instruction.i_out]
            block = (
                input_blocks[instruction.i_in]
                if input_blocks is not None
                else inputs[:, input_slices[instruction.i_in]]
            )
            if getattr(self, "input_layout", "mul_ir") == "ir_mul":
                block = block.reshape(-1, ir.dim, ci)
            else:
                block = block.reshape(-1, ci, ir.dim).transpose(1, 2)
            value = torch.matmul(block, matrix.reshape(ci, co))
            value = value * instruction.path_weight
            pieces[instruction.i_out].append(
                value.transpose(1, 2).reshape(inputs.shape[0], co * ir.dim)
            )
        outputs = []
        bias_offset = 0
        for (co, ir), terms in zip(self.irreps_out, pieces):
            if terms:
                value = terms[0]
                for term in terms[1:]:
                    value = value + term
            else:
                value = inputs.new_zeros((inputs.shape[0], co * ir.dim))
            if ir.is_scalar() and self.bias is not None:
                value = value + self.bias[bias_offset : bias_offset + co]
                bias_offset += co
            outputs.append(value)
        return torch.cat(outputs, dim=-1)


def grouped_networks(
    networks,
    updates,
    radial,
    source_types,
    target_types,
    *,
    hidden,
    embedding=None,
    basis=None,
):
    """Evaluate geometry-only MLPs with one batched matrix product per stage.

    Parameters and element embeddings remain differentiable runtime tensors.
    An independent group is used for each actual sequence of layer dimensions.
    """
    groups = {}
    outputs = [None] * len(networks)
    for index, network in enumerate(networks):
        layers = tuple(network.mlp)[:-1] if hidden else tuple(network.mlp)
        signature = tuple(
            (
                type(layer).__name__,
                getattr(layer, "in_dim", None),
                getattr(layer, "out_dim", None),
                getattr(layer, "eps", None),
                getattr(layer, "scale_factor", None),
            )
            for layer in layers
        )
        groups.setdefault(
            (signature, hasattr(updates[index], "source_embedding")), []
        ).append((index, layers))
    radial_width = radial.shape[-1]
    for (_, conditioned), entries in groups.items():
        ids = [index for index, _ in entries]
        if not entries[0][1]:
            for index in ids:
                outputs[index] = (
                    torch.cat(
                        (
                            radial,
                            updates[index].target_embedding.weight[target_types],
                            updates[index].source_embedding.weight[source_types],
                        ),
                        dim=-1,
                    )
                    if conditioned
                    else radial
                )
            continue
        first = [layers[0] for _, layers in entries]
        weights = torch.stack([layer.get_weight() for layer in first])
        radial_weight = weights[:, :radial_width]
        if embedding is None:
            values = torch.matmul(radial.unsqueeze(0), radial_weight)
        else:
            if type(embedding).__name__ != "LinearEdgeEmbedding":
                raise ValueError("Composition requires a linear edge embedding")
            projection = embedding.radial_proj
            embedding_weight = projection.scalar_matrix()
            combined = torch.matmul(embedding_weight.unsqueeze(0), radial_weight)
            values = torch.matmul(basis.unsqueeze(0), combined)
            if projection.bias is not None:
                values = values + torch.matmul(
                    projection.bias[None, None, :], radial_weight
                )
        if first[0].bias is not None:
            values = values + torch.stack([layer.bias for layer in first])[:, None, :]
        if conditioned:
            target_embedding = torch.stack(
                [updates[index].target_embedding.weight for index in ids]
            )
            source_embedding = torch.stack(
                [updates[index].source_embedding.weight for index in ids]
            )
            split = radial_width + target_embedding.shape[-1]
            values = (
                values
                + torch.bmm(target_embedding, weights[:, radial_width:split])[
                    :, target_types
                ]
            )
            values = (
                values
                + torch.bmm(source_embedding, weights[:, split:])[:, source_types]
            )
        for stage in range(1, len(entries[0][1])):
            layers = [sequence[stage] for _, sequence in entries]
            layer = layers[0]
            if type(layer).__name__ == "mlpLinear":
                values = torch.bmm(
                    values, torch.stack([item.get_weight() for item in layers])
                )
                if layer.bias is not None:
                    values = (
                        values + torch.stack([item.bias for item in layers])[:, None, :]
                    )
            elif isinstance(layer, torch.nn.LayerNorm):
                mean = values.mean(-1, keepdim=True)
                variance = (values - mean).square().mean(-1, keepdim=True)
                values = (values - mean) * torch.rsqrt(variance + layer.eps)
                if layer.weight is not None:
                    values = (
                        values
                        * torch.stack([item.weight for item in layers])[:, None, :]
                    )
                if layer.bias is not None:
                    values = (
                        values + torch.stack([item.bias for item in layers])[:, None, :]
                    )
            elif isinstance(layer, torch.nn.RMSNorm):
                eps = torch.finfo(values.dtype).eps if layer.eps is None else layer.eps
                values = values * torch.rsqrt(
                    values.square().mean(-1, keepdim=True) + eps
                )
                if layer.weight is not None:
                    values = (
                        values
                        * torch.stack([item.weight for item in layers])[:, None, :]
                    )
            else:
                if any(tuple(item.parameters()) for item in layers):
                    raise ValueError("Unsupported parameterized radial activation")
                values = layer(values)
        for index, value in zip(ids, values.unbind(0)):
            outputs[index] = value
    return outputs


def grouped_projections(
    networks, updates, hidden, radial, source_types, target_types, mode
):
    """Project all geometry-only layers together with live parameter matrices.

    Single-layer radial heads factor their shared radial input from compact
    element embeddings. Element projections are evaluated before edge gathers,
    and the 128-wide radial contraction stays aligned for Tensor Cores.
    """
    outputs, groups = [None] * len(networks), {}
    for index, network in enumerate(networks):
        layer = network.mlp[-1]
        key = (
            (layer.in_dim, layer.out_dim),
            layer.bias is not None,
            len(network.mlp) == 1 and hasattr(updates[index], "source_embedding"),
        )
        groups.setdefault(key, []).append(index)
    for (_, has_bias, direct), indices in groups.items():
        layers = [networks[i].mlp[-1] for i in indices]
        weight = torch.stack([layer.get_weight() for layer in layers])
        if direct:
            width = radial.shape[-1]
            target = torch.stack([updates[i].target_embedding.weight for i in indices])
            source = torch.stack([updates[i].source_embedding.weight for i in indices])
            channels = target.shape[-1]
            if weight.shape[1] != width + 2 * channels:
                raise ValueError("Unexpected direct radial conditioning width")
            if mode == "split":
                elements = torch.cat(
                    (target[:, target_types], source[:, source_types]), -1
                )
                conditioning = torch.bmm(elements, weight[:, width:])
                value = torch.baddbmm(
                    conditioning,
                    radial.unsqueeze(0).expand(len(indices), -1, -1),
                    weight[:, :width],
                )
            else:
                value = torch.matmul(radial.unsqueeze(0), weight[:, :width])
                value = (
                    value
                    + torch.bmm(target, weight[:, width : width + channels])[
                        :, target_types
                    ]
                )
                value = (
                    value
                    + torch.bmm(source, weight[:, width + channels :])[:, source_types]
                )
        else:
            value = torch.bmm(torch.stack([hidden[i] for i in indices]), weight)
        if has_bias:
            value = value + torch.stack([layer.bias for layer in layers])[:, None, :]
        for index, output in zip(indices, value.unbind(0)):
            outputs[index] = output
    return outputs


def project_radial(network, features, materialize, affine):
    """Return edge features and their shared input-by-output projection."""
    last = network.mlp[-1]
    weight = last.get_weight()
    if materialize and affine:
        result = (
            features @ weight
            if last.bias is None
            else torch.addmm(last.bias, features, weight)
        )
        return result, result.new_empty((0, result.shape[-1]))
    if last.bias is not None:
        features = torch.cat((features, torch.ones_like(features[:, :1])), -1)
        weight = torch.cat((weight, last.bias.unsqueeze(0)), 0)
    if materialize:
        features = features @ weight
        weight = features.new_empty((0, features.shape[-1]))
    return features, weight


def degree_projections(
    networks,
    updates,
    radial,
    source,
    target,
    mode,
    embedding,
    basis,
    materialize,
    jet=None,
):
    """Batch each harmonic degree across layers and restore convolution path order."""
    if (
        jet is not None
        and materialize
        and embedding is None
        and all(type(update).__name__ == "IdentityEdgeUpdate" for update in updates)
        and all(
            head.mlp[-1].bias is None
            for network in networks
            for head in network.branches
        )
    ):
        from .radial_projection import project_degree_jets

        return project_degree_jets(networks, radial, jet)
    parts = radial.split(networks[0].in_features, -1)
    basis_parts = basis.split(embedding.widths, -1) if embedding is not None else ()
    slopes = jet[1].split(networks[0].in_features, -1) if jet is not None else ()
    blocks = [[] for _ in networks]
    for degree, part in enumerate(parts):
        ids = [i for i, net in enumerate(networks) if degree in net.active]
        if not ids:
            continue
        heads = [networks[i].branches[networks[i].active.index(degree)] for i in ids]
        conditioning = [updates[i] for i in ids]
        if jet is None:
            hidden = grouped_networks(
                heads,
                conditioning,
                part,
                source,
                target,
                hidden=True,
                embedding=embedding.embeddings[degree]
                if embedding is not None
                else None,
                basis=basis_parts[degree] if embedding is not None else None,
            )
        else:
            from .radial_jet import AttachRadialJet, packed_hidden_jets

            values, tangents = packed_hidden_jets(heads, part.detach(), slopes[degree])
            hidden = [
                AttachRadialJet.apply(value, tangent, jet[0])
                for value, tangent in zip(values, tangents)
            ]
        values = (
            grouped_projections(heads, conditioning, hidden, part, source, target, mode)
            if materialize
            else hidden
        )
        for i, value in zip(ids, values):
            blocks[i].append(value)
    results = []
    for net, values in zip(networks, blocks):
        features = torch.cat(values, -1) if values else radial[:, :0]
        if materialize:
            if net.output_order is not None:
                features = permute(features, net.output_order, net.inverse_order)
            projection = features.new_empty((0, net.out_features))
        else:
            # The aligned kernel consumes a single shared projection matrix.
            heads = [branch.mlp[-1] for branch in net.branches]
            projection = torch.block_diag(*(head.get_weight() for head in heads))
            if heads and heads[0].bias is not None:
                features = F.pad(features, (0, 1), value=1.0)
                projection = torch.cat(
                    (projection, torch.cat([head.bias for head in heads])[None]), 0
                )
            if net.output_order is not None:
                projection = permute(projection, net.output_order, net.inverse_order)
        results.append((features, projection))
    return results


def add_element_weights(
    features, projection, table, endpoints, materialize, offset=None
):
    """Add the mean of the endpoint path weights."""
    source, target = endpoints
    left, right = table if isinstance(table, tuple) else (table, table)
    if materialize:
        if offset is None:
            offset = left[source] + right[target]
        return torch.add(features, offset.to(features), alpha=0.5), projection
    elements = torch.arange(left.size(0), device=source.device)
    a = (source[:, None] == elements).to(features)
    b = (target[:, None] == elements).to(features)
    if isinstance(table, tuple):
        lookup, weights = torch.cat((a, b), -1), torch.cat((left, right))
    else:
        lookup, weights = a + b, left
    return torch.cat((features, lookup * 0.5), -1), torch.cat(
        (projection, weights.to(projection))
    )


class NodeProduct(torch.nn.Module):
    """Node product preserving the original gate matrix and Muon parameter."""

    def forward(self, node_feats, node_type, sc, batch, preprojected=None):
        projected = self.linear_up(node_feats) if preprojected is None else preprojected
        if getattr(self, "layout_adjoints", False):
            from .layout_adjoint import ColumnSlices, Permutation

            projected = Permutation.apply(
                projected, self.factor_index, self.factor_inverse_index
            )
            widths = (self.num_hidden_channel,) + (self.factor_dim,) * self.num_inputs
            scalars, *factor_blocks = ColumnSlices.apply(projected, widths)
        else:
            projected = projected.index_select(-1, self.factor_index)
            scalars = projected[:, : self.num_hidden_channel]
            factors = projected[:, self.num_hidden_channel :].reshape(
                node_feats.shape[0], self.num_inputs, self.factor_dim
            )
            factor_blocks = tuple(factors[:, i] for i in range(self.num_inputs))
        affine = self.gate[0]
        effective_weight = affine.weight * affine.alpha
        scalar_weight = effective_weight[:, : self.num_hidden_channel]
        species_weight = effective_weight[:, self.num_hidden_channel :]
        species_values = F.linear(self.embedding.weight, species_weight)
        if self.compose_gate:
            # Only linear scalar maps are associated. Sigmoid and all original
            # parameter objects remain unchanged.
            gate = node_feats.new_zeros((node_feats.shape[0], scalar_weight.shape[0]))
            index = self.gate_scalar_columns
            for weight_index, start, width, out_width, factor in self.gate_scalar_paths:
                matrix = self.linear_up.weight[weight_index].reshape(width, out_width)
                combined = matrix[:, index] @ scalar_weight.T
                gate = gate + (node_feats[:, start : start + width] @ combined) * factor
            if self.linear_up.bias is not None:
                bias = self.linear_up.bias[self.gate_scalar_bias_start + index]
                gate = gate + F.linear(bias, scalar_weight)
            if affine.bias is not None:
                gate = gate + affine.bias
            gate = gate + species_values[node_type]
        else:
            gate = (
                F.linear(scalars, scalar_weight, affine.bias)
                + species_values[node_type]
            )
        gate = self.gate[1](gate)
        if getattr(self, "layout_adjoints", False):
            gate_blocks = ColumnSlices.apply(gate, self.gate_partition_widths)
        else:
            gate_blocks = tuple(gate[:, start:stop] for start, stop in self.gate_slices)
        corr = factor_blocks[0]
        first_gated = corr * gate_blocks[0][:, self.gate_index_1]
        output = (
            composed_linear(self.coefs[0], self.linear, first_gated, final_bias=True)
            if self.compose_products
            else self.coefs[0](first_gated)
        )
        for index, (product, linear) in enumerate(zip(self.aces, self.coefs[1:])):
            factor = factor_blocks[index + 1 if self.asymmetric else 0]
            gate_block = gate_blocks[index + 1]
            if node_feats.is_cuda and self.product_mode == "packed":
                metadata, output_dim = self.packed_products[index]
                operands = [corr, factor, gate_block, corr.new_empty((0, output_dim))]
                mapping = (0, 1, 2, 3)
                if not self.asymmetric:
                    operands = [corr, gate_block, corr.new_empty((0, output_dim))]
                    mapping = (0, 0, 1, 2)
                gated = packed_contract(
                    metadata,
                    repr(((mapping, 3, 0, 0),)),
                    operands,
                )[0]
                output = output + (
                    composed_linear(linear, self.linear, gated, packed=True)
                    if self.compose_products
                    else packed_linear(linear, gated)
                )
                continue
            if node_feats.is_cuda and self.fused_gate:
                metadata, output_dim = self.gated_products[index]
                gated = gated_product(metadata, corr, factor, gate_block, output_dim)
            elif node_feats.is_cuda:
                metadata, output_dim = self.eqx_products[index]
                corr = channel_product(
                    metadata,
                    repr((((0, 1, 2), 2, 0, 0),)),
                    [corr, factor, corr.new_empty((0, output_dim))],
                )[0]
                expanded = gate_block.index_select(
                    -1, getattr(self, f"gate_index_{index + 2}")
                )
                gated = corr * expanded
            else:
                if self.product_layout == "ir_mul":
                    corr = product(
                        corr.index_select(-1, self.factor_original_order),
                        factor.index_select(-1, self.factor_original_order),
                    )
                    corr = corr.index_select(-1, self.product_output_order)
                else:
                    corr = product(corr, factor)
                expanded = gate_block.index_select(
                    -1, getattr(self, f"gate_index_{index + 2}")
                )
                gated = corr * expanded
            output = output + (
                composed_linear(linear, self.linear, gated)
                if self.compose_products
                else linear(gated)
            )
        if not self.compose_products:
            output = self.linear(output)
        if sc is not None:
            output = torch.add(sc, output, alpha=getattr(self, "residual_scale", 1.0))
        return output


class Representation(torch.nn.Module):
    """Run geometry, the complete geometry-only stack, then sequential nodes."""

    def forward(self, data, graph):
        if graph.lmp:
            raise ValueError("TACE-OAM-XL-v2 currently implements training graphs")
        node_type = graph.node_type
        edge_index = data["edge_index"]
        radial_map = data.get("oam_radial_map", data.get("edge_pair_inverse"))
        if radial_map is None:
            radial_index, length = edge_index, graph.edge_length
        else:
            representatives = data[
                "oam_radial_representatives"
                if "oam_radial_map" in data
                else "edge_pair_rows"
            ]
            radial_index, length = (
                edge_index[:, representatives],
                graph.edge_length[representatives],
            )
        basis, cutoff = self.radial_basis(
            length, node_type, radial_index, self.atomic_numbers
        )
        basis = basis.to(self.model_dtype)
        direction = graph.edge_vector / graph.edge_length.clamp_min(1e-12)
        attrs = (
            self.o3_angular_basis(direction).to(self.model_dtype)
            if self.use_o3_angular_basis
            else None
        )
        if self.use_cutoff_scatter_norm:
            norm = self.interactions[0].scatter_norm
            message_cutoff = norm.edge_weights(
                cutoff if radial_map is None else cutoff[radial_map],
                edge_index[1],
                node_type.size(0),
            ).to(self.model_dtype)
        cutoff = cutoff.to(self.model_dtype) if cutoff is not None else None
        edge_cutoff = (
            cutoff if cutoff is None or radial_map is None else cutoff[radial_map]
        )
        if not self.use_cutoff_scatter_norm:
            message_cutoff = edge_cutoff
        local_frame = (
            self.o2_angular_basis.forward_packed(
                graph.edge_vector, method=self.frame_method
            ).to(self.model_dtype)
            if self.has_local_interactions
            or getattr(self.node_embedding, "use_wigner", False)
            else None
        )
        nodes = self.node_embedding(
            node_type,
            basis if radial_map is None else basis[radial_map],
            edge_index,
            attrs,
            edge_cutoff,
            local_frame,
            local_frame,
        )
        radial = self.edge_embedding(nodes, node_type, basis, radial_index, cutoff)
        frame = None
        if self.angular_mode == "aligned":
            frame = packed_frame(self.frame_source.frame, direction).to(
                self.model_dtype
            )
        native_attrs = None
        if (
            self.cg_layers
            and self.edge_mode in ("scheduled", "flashtp")
            and nodes.is_cuda
        ):
            # Match the existing FP64 harmonic-times-amplitude computation
            # before the final model-dtype cast, and share it across layers.
            from e3nn import o3

            conv = self.interactions[self.cg_layers[0]].rejector.eqx_tp
            amplitude = (
                nodes.new_ones((1, conv.amplitude_dim))
                if message_cutoff is None
                else message_cutoff.expand(-1, conv.amplitude_dim)
            )
            sections, offset = [], 0
            for mul, degree in conv.harmonic_entries:
                harmonic = o3.spherical_harmonics(
                    degree, direction, normalize=False, normalization=conv.normalization
                )
                sections.append(
                    (
                        harmonic.unsqueeze(-1)
                        * amplitude[:, None, offset : offset + mul]
                    ).flatten(1)
                )
                offset += mul
            native_attrs = torch.cat(sections, -1).to(nodes)
        source_types, target_types = (
            node_type[radial_index[0]],
            node_type[radial_index[1]],
        )
        compressed_radial = (
            radial_map is not None
            and self.edge_mode == "scheduled"
            and not self.has_local_interactions
            and getattr(self.edge_embedding, "table_names", ())
            != ("source_embedding", "target_embedding")
            and nodes.is_cuda
            and os.environ.get("EQX_XL_COMPRESSED_RADIAL", "0") == "1"
        )
        radial_jet = (
            not self.degree_radial
            and self.training
            and os.environ.get("EQX_XL_RADIAL_JET", "0") == "1"
        )
        if radial_jet and self.radial_projection != "individual":
            raise ValueError("Radial jets require individual heads")
        embedding = (
            self.edge_embedding
            if self.radial_projection == "composed"
            and hasattr(self.edge_embedding, "radial_proj")
            else None
        )
        networks = [layer.edge_info for layer in self.interactions]
        degree_heads = pair_rows = pairs = None
        degree_stream = (
            self.degree_radial
            and self.radial_mode == "stream"
            and self.angular_mode == "cg"
        )
        if degree_stream:
            from equivariantx.conv import EdgePairs, indexed_rows

            if radial_map is not None:
                pairs = EdgePairs(representatives, radial_map)
            if getattr(self.edge_embedding, "table_names", ()) == ("embedding",):
                pair_rows = indexed_rows(
                    node_type[radial_index],
                    self.num_elements,
                    dtype=nodes.dtype,
                    reduce="mean",
                    backend=self.interactions[0].rejector.eqx_tp.backend,
                )
        if self.degree_radial:
            if not degree_stream:
                jet = None
                if self.training and self.radial_jets:
                    from .radial_jet import jn_slope

                    jet = length, jn_slope(self.radial_basis, length).to(radial)
                degree_embedding = (
                    self.edge_embedding
                    if self.radial_projection == "composed"
                    and hasattr(self.edge_embedding, "embeddings")
                    and type(self.edge_embedding.embeddings[0]).__name__
                    == "LinearEdgeEmbedding"
                    else None
                )
                degree_heads = degree_projections(
                    networks,
                    self.edge_updates,
                    radial,
                    source_types,
                    target_types,
                    self.radial_projection,
                    degree_embedding,
                    basis,
                    self.radial_mode == "materialize",
                    jet=jet,
                )
        elif radial_jet:
            from .radial_jet import AttachRadialJet, hidden_jets

            values, slopes = hidden_jets(
                self, networks, length, node_type, radial_index
            )
            hidden = [
                AttachRadialJet.apply(value, slope, length)
                for value, slope in zip(values, slopes)
            ]
        else:
            hidden = grouped_networks(
                networks,
                self.edge_updates,
                radial,
                source_types,
                target_types,
                hidden=True,
                embedding=embedding,
                basis=basis,
            )
        projected = None
        if not self.degree_radial and self.radial_projection in ("batched", "split"):
            projected = grouped_projections(
                networks,
                self.edge_updates,
                hidden,
                radial,
                source_types,
                target_types,
                self.radial_projection,
            )
        density = None
        if not self.use_cutoff_scatter_norm:
            densities = grouped_networks(
                [layer.scatter_norm.edge_density for layer in self.interactions],
                self.edge_updates,
                radial,
                source_types,
                target_types,
                hidden=False,
                embedding=embedding,
                basis=basis,
            )
            edge_density = torch.cat(
                [torch.tanh(value.square()) for value in densities], dim=-1
            )
            if cutoff is not None:
                edge_density = edge_density * cutoff
            if radial_map is not None:
                edge_density = edge_density[radial_map]
            density = nodes.new_zeros((nodes.shape[0], self.num_layers)).index_add(
                0, edge_index[1], edge_density
            )
            density = density * torch.stack(
                [layer.scatter_norm.beta for layer in self.interactions]
            )
            density = density + torch.stack(
                [layer.scatter_norm.alpha for layer in self.interactions]
            )
        descriptors = []
        element_tables, element_offsets = {}, {}
        flash_indices = (
            edge_index.to(torch.int32) if self.edge_mode == "flashtp" else None
        )
        for index, (inter, product) in enumerate(zip(self.interactions, self.products)):
            packed_message = False
            mode = self.interaction_modes[index]
            res_bb = inter.resnetBB(nodes) if hasattr(inter, "resnetBB") else None
            res_ba = inter.resnetBA(nodes) if hasattr(inter, "resnetBA") else None
            associate_norm = os.environ.get("EQX_XL_ASSOCIATE_NORM", "0") == "1"
            if (
                associate_norm
                and hasattr(inter, "norm1")
                and not inter.norm1.center_scalars
            ):
                from .normalized_linear import normalized_linear

                nodes = normalized_linear(inter.linear_up, inter.norm1, nodes)
            else:
                if hasattr(inter, "norm1"):
                    nodes = inter.norm1(nodes)
                nodes = inter.linear_up(nodes)
            radial_network, edge_pairs = None, None
            if degree_stream:
                edge_pairs = pairs
                inputs = self.edge_updates[index](
                    nodes, node_type, radial, radial_index, cutoff
                )
                element = None
                if hasattr(self.edge_embedding, "layer_weights"):
                    table = self.edge_embedding.layer_weights(index)
                    element = (
                        table,
                        node_type[edge_index]
                        if isinstance(table, tuple)
                        else pair_rows,
                    )
                features, projection, radial_network = inter._radial_projection(
                    inputs, element, edge_pairs
                )
                if element is not None and isinstance(element[0], tuple):
                    edge_pairs = None
            elif degree_heads is not None:
                features, projection = degree_heads[index]
            elif projected is not None:
                features = projected[index]
                projection = features.new_empty((0, features.shape[-1]))
            else:
                features, projection = project_radial(
                    networks[index],
                    hidden[index],
                    self.radial_mode == "materialize",
                    self.affine_radial,
                )
            layer_radial_map = radial_map
            if not degree_stream and hasattr(self.edge_embedding, "layer_weights"):
                group = self.element_path_groups[index]
                if group not in element_tables:
                    element_tables[group] = self.edge_embedding.layer_weights(index)
                table = element_tables[group]
                endpoints = source_types, target_types
                if radial_map is not None and isinstance(table, tuple):
                    # Distance features are shared; directed element offsets are not.
                    features = features[radial_map]
                    layer_radial_map = None
                    endpoints = node_type[edge_index[0]], node_type[edge_index[1]]
                if self.radial_mode == "materialize" and group not in element_offsets:
                    left, right = table if isinstance(table, tuple) else (table, table)
                    element_offsets[group] = left[endpoints[0]] + right[endpoints[1]]
                features, projection = add_element_weights(
                    features,
                    projection,
                    table,
                    endpoints,
                    self.radial_mode == "materialize",
                    element_offsets.get(group),
                )
            if (
                layer_radial_map is not None
                and not compressed_radial
                and not degree_stream
            ):
                features = features[layer_radial_map]
            if mode in ("uv_o2", "uu_o2"):
                rejector = inter.rejector
                envelope = message_cutoff
                if envelope is None:
                    envelope = nodes.new_ones((1, 1))
                if mode == "uv_o2":
                    from tace.models.spherical.o2 import uv_convolution

                    radial_network = None
                    if projection.shape[0]:
                        features, radial_network = ConvolutionWeights(
                            features, projection, None
                        ).as_network()
                    message = uv_convolution(
                        rejector,
                        nodes,
                        None,
                        features,
                        edge_index,
                        local_frame,
                        None,
                        envelope,
                        radial_network=radial_network,
                    )
                else:
                    message = rejector.forward_stream(
                        nodes,
                        features,
                        projection,
                        edge_index,
                        local_frame,
                        envelope,
                        None,
                    )
            elif self.edge_mode == "flashtp":
                message = self.flash_convolutions[index](
                    nodes, native_attrs, features, flash_indices[0], flash_indices[1]
                )
            elif native_attrs is not None:
                from .edge import PACKED_EDGES, native_message

                packed_message = PACKED_EDGES
                message = native_message(
                    inter.rejector,
                    nodes,
                    native_attrs,
                    features,
                    edge_index,
                    packed=packed_message,
                    radial_map=radial_map if compressed_radial else None,
                )
            elif self.angular_mode == "aligned":
                aligned = self.aligned[index]
                amplitude = (
                    torch.ones_like(direction[:, :1])
                    if message_cutoff is None
                    else message_cutoff
                )
                message = aligned(
                    inter.rejector.reshape_in(nodes),
                    features,
                    projection,
                    frame,
                    amplitude.expand(-1, aligned.num_harmonics),
                    edge_index,
                    nodes.shape[0],
                )
                message = inter.rejector.reshape_out(message)
            else:
                message = inter.rejector.forward_stream(
                    nodes,
                    attrs,
                    features,
                    projection,
                    edge_index,
                    message_cutoff,
                    edge_vector=direction,
                    radial_network=radial_network,
                    edge_pairs=edge_pairs,
                )
            message = (
                packed_linear(inter.linear_down, message)
                if packed_message
                else inter.linear_down(message)
            )
            if density is not None:
                message = message / density[:, index : index + 1]
            message = inter.linear_nonlinearity(inter.nonlinearity(message))
            if res_ba is not None:
                message = torch.add(
                    res_ba, message, alpha=getattr(inter, "residual_scale", 1.0)
                )
            res_ab = inter.resnetAB(message) if hasattr(inter, "resnetAB") else None
            preprojected = None
            if (
                associate_norm
                and hasattr(inter, "norm2")
                and not inter.norm2.center_scalars
            ):
                if product.compose_gate:
                    raise ValueError(
                        "Associated normalization and composed scalar gate are separate experiments"
                    )
                from .normalized_linear import normalized_linear

                preprojected = normalized_linear(
                    product.linear_up, inter.norm2, message
                )
            elif hasattr(inter, "norm2"):
                message = inter.norm2(message)
            sc = res_bb if res_bb is not None else res_ab
            nodes = product(
                message, node_type, sc, data["batch"], preprojected=preprojected
            )
            if index == self.num_layers - 1 and hasattr(self, "final_norm"):
                nodes = self.final_norm(nodes)
            descriptors.append(nodes)
        return {
            "descriptors": descriptors,
            "uie_feats": None,
            "noise_mask_tensor": None,
            "dens_batch_mask_tensor": None,
            "magnetic_radial_basis": None,
        }


def validate(source):
    """Reject architectures outside the explicit OAM-XL execution contract."""
    representation = source.readout_fn.representation
    if source.lmp:
        raise ValueError("TACE-OAM-XL-v2 currently implements ordinary training graphs")
    if type(source.readout_fn).__name__ != "SphericalTACE":
        raise ValueError("TACE-OAM-XL-v2 requires SphericalTACE")
    if representation.use_dens or representation.use_magnetic_interaction:
        raise ValueError("Denoising and magnetic paths are not in the OAM-XL contract")
    if (
        hasattr(representation, "uie_embedding")
        or representation.uee_embeddings is not None
    ):
        raise ValueError("External-field embeddings are not in the OAM-XL contract")
    if source.readout_fn.model_config["long_range"]["les"]["enable"]:
        raise ValueError("Long-range LES is not in the OAM-XL contract")
    if set(source.readout_fn.target_property) - {
        "energy",
        "forces",
        "stress",
        "virials",
    }:
        raise ValueError("Only conservative E/F/S properties are supported")
    for module in source.modules():
        if (
            getattr(module, "use_lora", False)
            or type(module).__name__ == "GraphDropPath"
        ):
            raise ValueError(
                "LoRA and stochastic depth require another execution contract"
            )
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise ValueError("Dropout requires another execution contract")
    for inter, product, update in zip(
        representation.interactions,
        representation.products,
        representation.edge_updates,
    ):
        if (
            type(inter).__name__
            not in ("O3CgtpInteraction", "UvO2Interaction", "UuO2Interaction")
            or inter.resnet_linear_type != "identity"
        ):
            raise ValueError(
                "Expected cgtp, uv_o2 or uu_o2 interactions with identity residuals"
            )
        if type(inter).__name__ == "UvO2Interaction" and inter.rejector.eqx_tp is None:
            raise ValueError(
                "The UV O(2) gate activations must support fused execution"
            )
        if (
            type(inter).__name__ in ("UvO2Interaction", "UuO2Interaction")
            and source.readout_fn.model_config["radial_basis"]["apply_cutoff"]
        ):
            raise ValueError(
                "Local O(2) interactions require radial_basis.apply_cutoff=False"
            )
        if type(product).__name__ != "NonlinearCgtpACE" or product.correlation != 2:
            raise ValueError("Expected correlation-two nonlinear CGTP products")
        if type(update).__name__ not in ("IdentityEdgeUpdate", "Element2EdgeUpdate"):
            raise ValueError(
                "Expected identity or element2 geometry-only edge conditioning"
            )
        if type(inter.edge_info).__name__ not in ("MLPV2", "BatchMLPV2"):
            raise ValueError("Expected MLPV2 radial networks")
        if type(inter.scatter_norm).__name__ not in (
            "DensityScatterNorm",
            "CutoffScatterNorm",
        ):
            raise ValueError("Expected density or cutoff scatter normalization")
    require_native_messages(representation)
    return Architecture(
        representation.num_layers,
        representation.num_channel,
        representation.num_elements,
        sum(p.numel() for p in source.parameters()),
        tuple(name for name, _ in source.named_parameters()),
    )


class TACEOAMXLV2(torch.nn.Module):
    """Architecture-specialized model with a joint E/F/S training boundary.

    Parameters
    ----------
    adapter : torch.nn.Module
        Converted property adapter. Use :meth:`from_tace` to construct one.
    architecture : Architecture
        Validated static topology and parameter identity.

    Examples
    --------
    >>> # Conversion keeps the optimizer's original Parameter objects.
    >>> # xl = TACEOAMXLV2.from_tace(model)
    >>> # predictions, loss = xl.forward_loss(batch, loss_fn)
    >>> # loss.backward(inputs=tuple(xl.parameters()))
    """

    name = "TACE-OAM-XL-v2"

    def __init__(self, adapter, architecture):
        super().__init__()
        self.readout_fn = adapter.readout_fn
        object.__setattr__(self, "adapter", adapter)
        self.architecture = architecture
        self.train(adapter.training)

    @classmethod
    def from_tace(
        cls,
        model,
        *,
        fuse_products=False,
        radial_mode="materialize",
        angular_mode="cg",
        product_layout="ir_mul",
        linear_mode="torch",
        edge_mode="original",
        gradient_schedule="original",
        product_mode="original",
        radial_projection="individual",
        compose_products=False,
        affine_radial=False,
        compose_gate=False,
        padded_batches=False,
        radial_jets=False,
    ):
        """Convert module execution, sharing every original trainable parameter.

        Parameters
        ----------
        model : torch.nn.Module
            TACE property adapter with CGTP, UV O(2) or UU O(2) interactions.
            Move it to the desired device and dtype before conversion.
        fuse_products : bool, optional
            Include compact learned gates in the local CG product and its
            recursively generated mixed adjoints. Defaults to False because
            the additional mixed adjoints were slower in train-2 benchmarks.
        radial_mode : {"stream", "materialize"}, optional
            Keep bounded radial projection workspaces, or expose complete
            projections to the joint compiler for reuse. Defaults to materialize.
        angular_mode : {"cg", "aligned"}, optional
            Use sparse CG contractions or an exact O(2) restriction with one
            packed orthonormal frame shared by all layers. Defaults to cg.
        product_layout : {"mul_ir", "ir_mul"}, optional
            Retain the original node-product layout or lower every product and
            its mixed adjoints to contiguous channel storage. Defaults to ir_mul.
        linear_mode : {"torch", "native"}, optional
            Experimental grouped linear and mixed-adjoint programs. Native
            execution directly reads and writes the model's irrep layouts.
        edge_mode : {"original", "scheduled", "flashtp"}, optional
            Select the CGTP edge schedule. Scheduled CGTP layers use materialized
            radial weights and shared harmonics. Local O(2) layers use their own
            fused convolution and also accept streamed radial projections.
        gradient_schedule : {"original", "fused"}, optional
            Delay exclusively consumed mixed weight adjoints into the final
            native gradient program through an Inductor graph pass.
        product_mode : {"original", "packed"}, optional
            Fuse the local CG, compact gate and matrix-input packing. Packed
            execution requires ir_mul layout and a single quadratic product.
        radial_projection : {"individual", "batched", "split", "composed"}, optional
            Batch final geometry-only projections across layers and project
            direct element conditioning before gathering onto edges. Composed
            mode associates only the linear embedding with the first MLP
            projection, preserving every following activation and hidden layer.
        compose_products : bool, optional
            Associate consecutive product coefficient and output matrices
            before applying them to nodes. Original parameters remain live.
        compose_gate : bool, optional
            Associate the invariant projection with the gate's scalar matrix.
            This experimental schedule keeps all activations and parameters.
        affine_radial : bool, optional
            Keep the radial projection's bias as an addmm epilogue instead of
            augmenting its inner dimension with a constant-one channel.
        radial_jets : bool, optional
            Evaluate fixed jn radial MLP values and slopes together during
            E/F/S training. Preserves their parameter gradients, but not
            coordinate Hessians. Disabled for evaluation. Defaults to False.

        Returns
        -------
        TACEOAMXLV2
            Independent execution modules with the original parameter objects
            and checkpoint names. The original optimizer remains valid.

        Notes
        -----
        Module registrations and buffers are copied. Parameters are shared:
        construct separate source models for independently updated references.
        Aligned execution requires nonzero edge vectors. All modes retain live
        weights and evaluate forces and stress from the same scalar energy.
        """
        from equivariantx_foundation.registry import require_cuda

        require_cuda(model)
        if radial_mode not in ("stream", "materialize"):
            raise ValueError("radial_mode must be stream or materialize")
        if angular_mode not in ("cg", "aligned"):
            raise ValueError("angular_mode must be cg or aligned")
        if product_layout not in ("mul_ir", "ir_mul"):
            raise ValueError("product_layout must be mul_ir or ir_mul")
        if linear_mode not in ("torch", "native"):
            raise ValueError("linear_mode must be torch or native")
        if edge_mode not in ("original", "scheduled", "flashtp"):
            raise ValueError("edge_mode must be original, scheduled or flashtp")
        if gradient_schedule not in ("original", "fused"):
            raise ValueError("gradient_schedule must be original or fused")
        if product_mode not in ("original", "packed"):
            raise ValueError("product_mode must be original or packed")
        if product_mode == "packed" and product_layout != "ir_mul":
            raise ValueError("Packed products require ir_mul layout")
        if radial_projection not in ("individual", "batched", "split", "composed"):
            raise ValueError("Unsupported radial_projection mode")
        if radial_projection != "individual" and radial_mode != "materialize":
            raise ValueError("Batched projections require materialized radial weights")
        from .precision import require_tensors, validate_environment

        validate_environment()
        require_tensors((*model.parameters(), *model.buffers()))
        architecture = validate(model)
        modes = {
            "O3CgtpInteraction": "cgtp",
            "UvO2Interaction": "uv_o2",
            "UuO2Interaction": "uu_o2",
        }
        interaction_modes = tuple(
            modes[type(layer).__name__]
            for layer in model.readout_fn.representation.interactions
        )
        if (
            "cgtp" in interaction_modes
            and edge_mode in ("scheduled", "flashtp")
            and (radial_mode != "materialize" or angular_mode != "cg")
        ):
            raise ValueError(
                "Scheduled CGTP edges require materialized radial weights and angular_mode='cg'"
            )
        if any(mode != "cgtp" for mode in interaction_modes) and (
            angular_mode == "aligned" or edge_mode == "flashtp"
        ):
            raise ValueError(
                "Aligned CGTP and FlashTP require CGTP interactions in every layer"
            )
        memo = {id(parameter): parameter for parameter in model.parameters()}
        readout = copy.deepcopy(model.readout_fn, memo)
        if padded_batches:
            from .padding import enable_padded_readout

            enable_padded_readout(readout)
        # Included in the adapter's compiled module contract and artifact key.
        readout.oam_xl_gradient_schedule = gradient_schedule
        persistent = set(model.readout_fn.state_dict())
        # FX deepcopy can register generated constants as persistent buffers.
        # Their execution values are retained, but checkpoint keys stay original.
        for path, child in readout.named_modules():
            for name in child._buffers.keys():
                key = f"{path}.{name}" if path else name
                if key not in persistent:
                    child._non_persistent_buffers_set.add(name)

        def lower(parent):
            for name, child in tuple(parent.named_children()):
                lower(child)
                if type(child).__name__ == "e3nnLinear":
                    linear = transplant(LinearMap, child)
                    instructions = child.linear.instructions
                    supported = (
                        isinstance(child.weight, torch.nn.ParameterList)
                        and len(instructions)
                        == len(child.irreps_in)
                        == len(child.irreps_out)
                        and len({i.i_in for i in instructions}) == len(instructions)
                        and len({i.i_out for i in instructions}) == len(instructions)
                    )
                    linear.native_linear = linear_mode == "native" and supported
                    setattr(parent, name, linear)

        lower(readout)
        representation = transplant(Representation, readout.representation)
        representation.degree_radial = representation.radial_basis.widths is not None
        representation.radial_mode = radial_mode
        representation.affine_radial = affine_radial
        representation.radial_projection = radial_projection
        representation.radial_jets = radial_jets
        if radial_jets:
            from .radial_jet import supports_degree_jets

            if radial_mode != "materialize" or not supports_degree_jets(representation):
                raise ValueError(
                    "Radial jets require fixed jn bases and matching scalar MLPs"
                )
        representation.angular_mode = angular_mode
        representation.edge_mode = edge_mode
        representation.interaction_modes = interaction_modes
        representation.element_path_groups = tuple(range(representation.num_layers))
        embedding = representation.edge_embedding
        if hasattr(embedding, "layer_weights"):
            groups, representatives = [], {}
            for layer in range(representation.num_layers):
                index = getattr(embedding, f"path_index_{layer}")
                key = None if index is None else tuple(index.cpu().tolist())
                groups.append(representatives.setdefault(key, layer))
            representation.element_path_groups = tuple(groups)
        representation.cg_layers = tuple(
            i
            for i, mode in enumerate(representation.interaction_modes)
            if mode == "cgtp"
        )
        representation.has_local_interactions = (
            len(representation.cg_layers) != representation.num_layers
        )
        representation.frame_method = (
            "generator"
            if representation.has_local_interactions
            and representation.o2_angular_basis.lmax >= 4
            else "auto"
        )
        if representation.cg_layers and edge_mode in ("scheduled", "flashtp"):
            convolutions = [
                representation.interactions[i].rejector.eqx_tp
                for i in representation.cg_layers
            ]
            signatures = {
                (conv.harmonic_entries, conv.normalization, conv.normalize)
                for conv in convolutions
            }
            if len(signatures) != 1 or convolutions[0].normalize:
                raise ValueError(
                    "Scheduled edges require shared regular harmonic entries"
                )
        if edge_mode == "flashtp":
            from .flash import FlashConvolution

            parameter = next(model.parameters())
            representation.flash_convolutions = torch.nn.ModuleList(
                [
                    FlashConvolution(
                        inter.rejector.eqx_tp.kernel_metadata,
                        parameter.device,
                        parameter.dtype,
                    )
                    for inter in representation.interactions
                ]
            )
        if angular_mode == "aligned":
            device = next(model.parameters()).device
            representation.aligned = torch.nn.ModuleList(
                [
                    aligned_convolution(inter.rejector.tp).to(device=device)
                    for inter in representation.interactions
                ]
            )
            largest = max(representation.aligned, key=lambda item: item.frame.lmax)
            # Refer to an already registered module without duplicate state.
            object.__setattr__(representation, "frame_source", largest)
            for item in representation.aligned.modules():
                item._non_persistent_buffers_set.update(item._buffers.keys())
        representation.products = torch.nn.ModuleList(
            [transplant(NodeProduct, product) for product in representation.products]
        )
        for product in representation.products:
            if not hasattr(product, "eqx_products"):
                # A native host has no optional fusion metadata. XL consumes
                # just the CG contractions, not the host's fused gate program.
                from ..nonlinear_ace import build_product

                product.eqx_products = tuple(
                    (build_product(ace.tp), ace.irreps_out.dim) for ace in product.aces
                )
            product.fused_gate = fuse_products
            product.product_layout = product_layout
            product.product_mode = product_mode
            product.compose_products = compose_products
            product.compose_gate = compose_gate
            if compose_gate:
                scalar_columns = product.factor_index[
                    : product.num_hidden_channel
                ].cpu()
                output_slices = product.linear_up.irreps_out.slices()
                candidates = [
                    i
                    for i, ((channels, ir), sl) in enumerate(
                        zip(product.linear_up.irreps_out, output_slices)
                    )
                    if ir.is_scalar()
                    and bool(
                        (
                            (scalar_columns >= sl.start) & (scalar_columns < sl.stop)
                        ).all()
                    )
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Scalar gate composition needs one scalar output block"
                    )
                output_index = candidates[0]
                product.register_buffer(
                    "gate_scalar_columns",
                    (scalar_columns - output_slices[output_index].start).to(
                        product.factor_index.device
                    ),
                    persistent=False,
                )
                product.gate_scalar_paths = []
                for wi, instruction in enumerate(product.linear_up.linear.instructions):
                    if instruction.i_out != output_index:
                        continue
                    ci, ir = product.linear_up.irreps_in[instruction.i_in]
                    if not ir.is_scalar():
                        raise ValueError(
                            "Gate scalar projection must have scalar inputs"
                        )
                    co, _ = product.linear_up.irreps_out[output_index]
                    product.gate_scalar_paths.append(
                        (
                            wi,
                            product.linear_up.irreps_in.slices()[
                                instruction.i_in
                            ].start,
                            ci,
                            co,
                            instruction.path_weight,
                        )
                    )
                product.gate_scalar_bias_start = sum(
                    c
                    for c, ir in product.linear_up.irreps_out[:output_index]
                    if ir.is_scalar()
                )
            if product_mode == "packed" and len(product.aces) != 1:
                raise ValueError("Packed products currently require correlation two")
            product.packed_products = []
            if product_layout == "ir_mul":
                factor_order = ir_mul_order(product.coefs[0].irreps_in)
                output_order = ir_mul_order(product.coefs[1].irreps_in)
                device = product.factor_index.device
                index = torch.cat(
                    (
                        torch.arange(product.num_hidden_channel),
                        *(
                            product.num_hidden_channel
                            + i * product.factor_dim
                            + factor_order
                            for i in range(product.num_inputs)
                        ),
                    )
                ).to(device)
                product.factor_index = product.factor_index[index]
                product.gate_index_1 = product.gate_index_1[factor_order.to(device)]
                product.gate_index_2 = product.gate_index_2[output_order.to(device)]
                metadata, width = product.eqx_products[0]
                if product_mode == "packed":
                    start, stop = product.gate_slices[1]
                    blocks = [
                        mul * ir.dim // product.num_hidden_channel
                        for mul, ir in product.coefs[1].irreps_in
                    ]
                    angular = angular_metadata(
                        product.aces[0].tp,
                        product.gate_index_2,
                        stop - start,
                        blocks,
                        product.num_hidden_channel,
                    )
                    if angular is not None:
                        product.packed_products.append((angular, width))
                if product.packed_products:
                    product.eqx_products = ()
                else:
                    metadata = channel_contiguous_metadata(
                        metadata,
                        (factor_order, factor_order, output_order),
                        product.num_hidden_channel,
                    )
                    product.eqx_products = ((metadata, width),)
                for linear in product.coefs:
                    linear.input_layout = "ir_mul"
                product.register_buffer(
                    "factor_original_order",
                    factor_order.argsort().to(device),
                    persistent=False,
                )
                product.register_buffer(
                    "product_output_order", output_order.to(device), persistent=False
                )
            product.layout_adjoints = (
                os.environ.get("EQX_XL_LAYOUT_ADJOINTS", "0") == "1"
            )
            if product.layout_adjoints:
                index = product.factor_index
                columns = product.linear_up.irreps_out.dim
                if index.numel() != columns or not torch.equal(
                    index.sort().values, torch.arange(columns, device=index.device)
                ):
                    raise ValueError("The factor layout must be a complete permutation")
                product.register_buffer(
                    "factor_inverse_index", index.argsort(), persistent=False
                )
                end = 0
                widths = []
                for start, stop in product.gate_slices:
                    if start != end or stop < start:
                        raise ValueError(
                            "Gate blocks must partition the complete output"
                        )
                    widths.append(stop - start)
                    end = stop
                if end != product.gate[0].weight.shape[0]:
                    raise ValueError("Gate partition does not match its affine output")
                product.gate_partition_widths = tuple(widths)
            product.gated_products = []
            for index, (metadata, width) in enumerate(product.eqx_products):
                start, stop = product.gate_slices[index + 1]
                mapping = getattr(product, f"gate_index_{index + 2}").cpu().tolist()
                if product_mode == "packed":
                    blocks = [
                        mul * ir.dim // product.num_hidden_channel
                        for mul, ir in product.coefs[index + 1].irreps_in
                    ]
                    product.packed_products.append(
                        (
                            packed_metadata(
                                metadata,
                                mapping,
                                stop - start,
                                blocks,
                            ),
                            width,
                        )
                    )
                elif fuse_products:
                    product.gated_products.append(
                        (gated_product_metadata(metadata, mapping, stop - start), width)
                    )
        readout.representation = representation
        from tace.models.compile.wrapper import CompileTensorModel

        from .adapter import OAMCompileAdapter

        adapter = (
            OAMCompileAdapter(readout)
            if isinstance(model, CompileTensorModel)
            else type(model)(readout)
        )
        adapter.reset_fidelity_idx(model.get_fidelity_idx())
        adapter.train(model.training)
        result = cls(adapter, architecture)
        before = dict(model.named_parameters())
        after = dict(result.named_parameters())
        if before.keys() != after.keys() or any(
            before[name] is not after[name] for name in before
        ):
            raise RuntimeError(
                "Conversion changed parameter identity or checkpoint names"
            )
        return result

    def train(self, mode=True):
        super().train(mode)
        self.adapter.train(mode)
        return self

    def _apply(self, fn, recurse=True):
        # The adapter owns precision bookkeeping and compiled-cache invalidation.
        self.adapter._apply(fn, recurse)
        from .precision import require_tensors

        require_tensors((*self.parameters(), *self.buffers()))
        return self

    def get_fidelity_idx(self):
        return self.adapter.get_fidelity_idx()

    def reset_fidelity_idx(self, fidelity_idx=0):
        return self.adapter.reset_fidelity_idx(fidelity_idx)

    def get_target_property(self):
        return self.adapter.get_target_property()

    def prepare_property_data(self, data):
        return self.adapter.prepare_property_data(data)

    def forward(self, data):
        from .precision import validate_call

        validate_call(data)
        return self.adapter(data)

    def forward_loss(self, data, loss_fn):
        """Evaluate the complete joint training program for the original loss."""
        from .precision import validate_call

        validate_call(data)
        from .extensions import extension_identity

        extension_identity()
        if hasattr(self.adapter, "forward_loss"):
            if self.readout_fn.oam_xl_gradient_schedule == "fused":
                from torch._inductor import config

                from .program import MixedGradientPass

                passes = {"post_grad_custom_pre_pass": MixedGradientPass()}
                if os.environ.get("EQX_XL_PARALLEL_MATRICES", "0") == "1":
                    # The joint graph is already differentiated. Match its
                    # parameter reductions before Inductor normalizes them.
                    passes["pre_grad_custom_pass"] = MixedGradientPass()
                with config.patch(passes):
                    return self.adapter.forward_loss(data, loss_fn)
            return self.adapter.forward_loss(data, loss_fn)
        data = self.prepare_property_data(data)
        predictions = self.adapter(data)
        return predictions, loss_fn(predictions, data)
