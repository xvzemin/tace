"""Local-order matrix products for the released rotary-attention architecture."""

import torch

from equivariantx.nn import graph_softmax
from equivariantx.ops.common.channel_product import local_product
from equivariantx.ops.common.frame import contract_cuda
from equivariantx.ops.common.polynomial import polynomial_adjoint
from equivariantx.utils.autograd import forward_ad_active
from equivariantx.utils.outputs import EDGE, MATRIX, NODE
from tace.models.spherical.inter import O3CgtpInteraction
from tace.models.spherical.tece_oam_rra import BilinearMoEACE, UvSO2Interaction

from ..foundation import shared_parameter_copy
from ..matching import plain_element_lookups, require_native_messages


class IndexedFrame(torch.autograd.Function):
    """Use the indexed CUDA contraction for each role of a released frame."""

    @staticmethod
    def forward(nodes, matrix, edges, index, count, layout, role):
        return contract_cuda(nodes, matrix, edges, index, count, layout, role)

    @staticmethod
    def setup_context(ctx, inputs, output):
        nodes, matrix, edges, index, ctx.count, ctx.layout, ctx.role = inputs
        ctx.save_for_backward(nodes, matrix, edges, index)

    @staticmethod
    def backward(ctx, grad):
        nodes, matrix, edges, index = ctx.saved_tensors
        values = [nodes, matrix, edges]
        values[ctx.role] = grad
        result = [
            contract_cuda(*values, index, ctx.count, ctx.layout, role)
            if role != ctx.role and ctx.needs_input_grad[role]
            else None
            for role in (NODE, MATRIX, EDGE)
        ]
        return *result, None, None, None, None


class EdgeExpansion(torch.autograd.Function):
    """Contract the edge expansion or its transposes with one CUDA path table."""

    @staticmethod
    def forward(metadata, program, *operands):
        return tuple(
            torch.ops.eqx.local_channel_product(metadata, program, list(operands))
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.description, ctx.program, *operands = inputs
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*operands)

    @staticmethod
    def backward(ctx, *gradients):
        operands = ctx.saved_tensors
        program, values, destinations = polynomial_adjoint(
            ctx.program, operands, gradients, ctx.needs_input_grad[2:]
        )
        dx = [None] * len(operands)
        if destinations:
            result = EdgeExpansion.apply(ctx.description, program, *values)
            for i, slot in destinations.items():
                dx[i] = result[slot]
        return None, None, *dx


def edge_expansion(metadata, packed, gates, coefficients, scalars):
    """Combine the residual, gated features and weighted edge products."""
    if not packed.is_cuda or forward_ad_active():
        return local_product(metadata, packed, packed, gates, coefficients, scalars)
    return EdgeExpansion.apply(
        metadata,
        "(((0,0,1,2,3,4),5,0,0),)",
        packed,
        gates,
        coefficients,
        scalars,
        packed.new_empty((0, packed.shape[1])),
    )[0]


def rotation_layout(order, lmax, channels):
    inverse = order.argsort().tolist()
    blocks, offset = [], 0
    for ell in range(lmax + 1):
        d = 2 * ell + 1
        rows = tuple((inverse[ell**2 + j] * channels, j, 1) for j in range(d))
        blocks.append((ell**2 * channels, channels, d, offset, rows))
        offset += d * d
    width = (lmax + 1) ** 2 * channels
    return repr((width, width, offset, tuple(blocks)))


class OrderLinear(torch.nn.Module):
    """Apply one shared channel matrix to the real components of each order."""

    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.irreps_out = linear.irreps_out
        self.paths = []
        for (i, j, (ni, no), scale), (offset, _) in zip(
            linear._weight_instructions, linear._weight_offsets, strict=True
        ):
            if i != j or i != len(self.paths):
                raise ValueError("Ordered diagonal O(2) instructions are required.")
            dim = linear.irreps_in[i].ir.dim
            self.paths.append(
                (linear._input_slices[i].start, dim, ni, no, offset, scale)
            )

    def forward(self, x):
        outputs = []
        for start, dim, ni, no, offset, scale in self.paths:
            source = x[:, start : start + dim * ni].reshape(-1, ni)
            weight = self.weight[offset : offset + ni * no].reshape(ni, no)
            bias = self.bias if dim == 1 and self.bias.numel() else None
            product = torch.nn.functional.linear(source, (weight * scale).T, bias)
            outputs.append(product.reshape(x.shape[0], dim * no))
        return torch.cat(outputs, dim=1)


class RRAInteraction(UvSO2Interaction):
    """Evaluate the rotary score and local product using the same aligned pair."""

    frozen_expansion_scales = None
    training_expansion_scales = None

    @property
    def use_eqx(self):
        return True

    def _compute_messages(
        self,
        node_feats,
        node_types,
        edge_radial_basis,
        edge_feats,
        edge_attrs,
        edge_index,
        edge_cutoff,
        edge_wigner=None,
        edge_wigner_inv=None,
        magnetic_node_info=None,
        magnetic_edge_attrs=None,
        graph=None,
        projected_weights=None,
    ):
        conv = self.rejector
        if not node_feats.is_cuda:
            return super()._compute_messages(
                node_feats,
                node_types,
                edge_radial_basis,
                edge_feats,
                edge_attrs,
                edge_index,
                edge_cutoff,
                edge_wigner,
                edge_wigner_inv,
                magnetic_node_info,
                magnetic_edge_attrs,
                graph,
            )
        e = edge_index.shape[1]
        angular = (conv.lmax + 1) ** 2
        x = conv.reshape_in(node_feats)
        indexed = hasattr(self, "frame_layout") and not forward_ad_active()
        if indexed:
            x = x.flatten(1)
            empty = x.new_empty((0, x.shape[1]))
            source, target = (
                IndexedFrame.apply(
                    x, edge_wigner, empty, i, x.shape[0], self.frame_layout, EDGE
                ).reshape(e, angular, conv.num_channel)
                for i in (edge_index[0], edge_index[1])
            )
            local = torch.cat((source, target), dim=2)
        else:
            paired = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=2)
            rotations = []
            aligned = []
            cursor = 0
            for ell in range(conv.lmax + 1):
                d = 2 * ell + 1
                rotation = edge_wigner[:, cursor : cursor + d * d].reshape(e, d, d)
                rotations.append(rotation)
                part = paired[:, ell**2 : (ell + 1) ** 2]
                aligned.append(torch.bmm(rotation, part) if ell else part)
                cursor += d * d
            local = torch.cat(aligned, dim=1).index_select(1, self.order_index)
            source, target = local.split(conv.num_channel, dim=2)
        key = conv.key_proj(source.flatten(1)).reshape_as(source)
        query = conv.query_proj(target.flatten(1)).reshape_as(target)
        score = conv.rotary_score(query, key, edge_radial_basis, fused=True)
        attention = graph_softmax(
            score,
            edge_index[1],
            node_feats.shape[0],
            edge_cutoff,
            conv.attention_eps,
            fused=True,
        )
        if edge_cutoff is not None:
            attention = attention * edge_cutoff
        weights = (
            self.edge_info(edge_feats)
            if projected_weights is None
            else projected_weights
        )
        message = local_product(conv._eqx_weight_metadata, local, weights)
        packed = conv.linear_up(message)
        coefficients = conv.linear_coefs(message[:, : conv.irreps_in[0].mul])
        scales = (
            self.training_expansion_scales
            if self.training
            else self.frozen_expansion_scales
        )
        if indexed and scales is not None and packed.dtype == torch.float32:
            from .force import indexed_expansion

            message = indexed_expansion(
                conv._eqx_update_metadata,
                scales,
                packed,
                coefficients,
            )
        else:
            (g, ng), (s, ns) = conv.projection_slices
            message = edge_expansion(
                conv._eqx_update_metadata,
                packed,
                conv.tensor_act(packed.narrow(1, g, ng)),
                conv.scalar_act(coefficients),
                conv.scalar_act(packed.narrow(1, s, ns)),
            )
        message = conv.linear_down(message).reshape(
            e, angular, conv.num_head, conv.num_channel_per_head
        )
        message = (message * attention[:, None, :, None]).flatten(2)
        if indexed:
            total = IndexedFrame.apply(
                empty,
                edge_wigner,
                message.flatten(1),
                edge_index[1],
                x.shape[0],
                self.frame_layout,
                NODE,
            )
            return conv.reshape_out.inverse(
                total.reshape(x.shape[0], angular, conv.num_channel)
            )
        message = message.index_select(1, self.global_index)
        output = torch.cat(
            [
                torch.bmm(matrix.transpose(1, 2), message[:, ell**2 : (ell + 1) ** 2])
                if ell
                else message[:, :1]
                for ell, matrix in enumerate(rotations)
            ],
            dim=1,
        )
        aggregated = output.new_zeros(
            node_feats.shape[0], output.shape[1], conv.num_channel
        )
        return conv.reshape_out.inverse(aggregated.index_add(0, edge_index[1], output))


def expansion_scales(update):
    from equivariantx.utils.metadata import parse_metadata

    from .program import activation

    tensor, scalar = activation(update.tensor_act), activation(update.scalar_act)
    metadata = getattr(update, "_eqx_update_metadata", None)
    if (
        metadata is not None
        and getattr(update, "projection_slices", None) == ((0, 640), (640, 160))
        and parse_metadata(metadata)[:2] == (32, (110, 110, 20, 95, 5, 45))
        and tensor is not None
        and scalar is not None
        and tensor[0] == scalar[0] == "silu"
    ):
        return tensor[1], scalar[1], scalar[1]
    return None


def enable_frozen_expansion(model):
    """Enable the joint edge expansion on supported frozen FP32 interactions."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError(
            "The indexed force expansion requires a frozen evaluation model."
        )
    for layer in model.readout_fn.representation.interactions[1::2]:
        if not isinstance(layer, RRAInteraction):
            continue
        layer.frozen_expansion_scales = expansion_scales(layer.rejector)


def validate_rra(model):
    """Validate the alternating CGTP and full-order rotary-attention architecture."""
    from tace.models.linear import has_lora
    from tace.models.spherical.edge import Element2EdgeUpdate, NonLinearEdgeEmbedding
    from tace.models.spherical.node import LinearNodeEmbedding

    from .radial import RRARepresentation

    rep = model.readout_fn.representation
    if not (
        len(rep.interactions) == len(rep.products) == len(rep.edge_updates) == 8
        and type(rep.node_embedding) is LinearNodeEmbedding
        and type(rep.edge_embedding) is NonLinearEdgeEmbedding
        and all(type(edge) is Element2EdgeUpdate for edge in rep.edge_updates)
        and not any(has_lora(module) for module in rep.modules())
        and not rep.use_dens
        and not rep.use_magnetic_interaction
        and not hasattr(rep, "uie_embedding")
        and rep.uee_embeddings is None
        and rep.o2_angular_basis.lmax == rep.o2_angular_basis.mmax == 4
        and not model.readout_fn.use_alllayer
        and all(
            (
                type(inter) is O3CgtpInteraction
                if i % 2 == 0
                else isinstance(inter, UvSO2Interaction)
                and inter.use_asymmetric_contraction
                and inter.mmax == inter.lmax == 4
                and inter.num_head == 4
            )
            and inter.num_channel == 64
            and inter.resnet_type == "BAB"
            and inter.resnet_linear_type == "identity"
            and not hasattr(inter, "resnetBB")
            and isinstance(prod, BilinearMoEACE)
            and prod.use_bilinear_gate
            and prod.correlation == 2
            and prod.num_expert == 8
            and prod.use_shared_expert
            for i, (inter, prod) in enumerate(zip(rep.interactions, rep.products))
        )
    ):
        raise ValueError("Expected the eight-layer TECE-RRA-1.0 architecture.")
    if isinstance(rep, RRARepresentation) and not plain_element_lookups(rep):
        raise ValueError(
            "The installed TECE radial schedule requires plain node and endpoint embedding lookups."
        )
    require_native_messages(rep, installed=((RRAInteraction, UvSO2Interaction),))


def optimize_rra(model):
    """Return a rotary-attention model sharing the checkpoint parameters."""
    from .radial import RRARepresentation

    validate_rra(model)
    from equivariantx_foundation.registry import require_cuda

    require_cuda(model)
    if all(
        type(layer) is RRAInteraction
        for layer in model.readout_fn.representation.interactions[1::2]
    ):
        return model
    result = shared_parameter_copy(model)
    rep = result.readout_fn.representation
    if RRARepresentation.supports(rep):
        rep.__class__ = RRARepresentation
    order = rep.o2_angular_basis.local_indices
    inverse = order.argsort()
    first = rep.interactions[1]
    layout = rotation_layout(order, first.lmax, first.rejector.num_channel)
    for inter in rep.interactions[1::2]:
        inter.rejector.prepare_eqx()
        inter.__class__ = RRAInteraction
        inter.training_expansion_scales = expansion_scales(inter.rejector)
        inter.register_buffer("order_index", order, persistent=False)
        inter.register_buffer("global_index", inverse, persistent=False)
        inter.frame_layout = layout
        for name in (
            "linear_up",
            "linear_down",
            "linear_coefs",
            "key_proj",
            "query_proj",
            "radial_proj",
        ):
            setattr(inter.rejector, name, OrderLinear(getattr(inter.rejector, name)))
    return result
