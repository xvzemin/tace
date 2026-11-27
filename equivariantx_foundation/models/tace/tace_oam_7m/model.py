"""Element-conditioned radial heads and node updates of TACE-7M."""

import torch
from torch.nn import functional as F

from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
from equivariantx_foundation.models.tace.matching import (
    plain_element_lookups,
    require_native_messages,
)
from equivariantx_foundation.models.tace.tace_oam_l.convolution import bias_convolution
from tace.models.linear import has_lora
from tace.models.spherical.representation import NodeRepresentation


def radial_parameters(inter, update):
    info = inter.edge_info.mlp
    density = inter.scatter_norm.edge_density.mlp
    edge, target, source = torch.cat(
        (info[0].get_weight(), density[0].get_weight()), dim=1
    ).split(update.source_embedding.embedding_dim, dim=0)
    return (
        edge,
        update.source_embedding.weight @ source,
        update.target_embedding.weight @ target,
        torch.cat((info[0].bias, density[0].bias)),
    )


def radial_heads(inter, params, radial, source, target):
    edge, src_table, dst_table, bias = params
    pre = torch.addmm(bias, radial, edge) + dst_table[target] + src_table[source]
    return radial_activations(inter, pre)


def project_radial_heads(radial, heads, source_types, target_types):
    """Project edge features for several layers and add both element offsets.

    Examples
    --------
    Two reversed edges have identical radial features. Their outputs differ
    because each head looks up the source and target in separate tables.

    >>> radial = torch.ones(2, 1)
    >>> source = torch.tensor([[0.], [1.]])
    >>> target = torch.tensor([[0.], [10.]])
    >>> head = (torch.ones(1, 1), source, target, torch.zeros(1))
    >>> src, dst = torch.tensor([0, 1]), torch.tensor([1, 0])
    >>> (out,) = project_radial_heads(radial, [head], src, dst)
    >>> out.tolist()
    [[11.0], [2.0]]
    """
    weights, source, target, bias = zip(*heads)
    terms = torch.addmm(torch.cat(bias), radial, torch.cat(weights, dim=1))
    terms = terms + torch.cat(target, dim=1)[target_types]
    terms = terms + torch.cat(source, dim=1)[source_types]
    return terms.split([b.numel() for b in bias], dim=1)


def radial_activations(inter, pre):
    h, d = pre.split(64, dim=1)
    layers = inter.edge_info.mlp
    for j in (1, 2, 3, 4, 5):
        h = layers[j](h)
    density = inter.scatter_norm.edge_density.mlp
    d = density[3](density[2](density[1](d)))
    return h, d.square().tanh()


def geometry(rep, vector, length, types, index):
    basis, cutoff = rep.radial_basis(length, types, index, rep.atomic_numbers)
    cutoff = cutoff.to(rep.model_dtype)
    feat = rep.edge_embedding(None, types, basis.to(rep.model_dtype), index, None)
    attrs = rep.o3_angular_basis(vector / length.clamp_min(1e-12)).to(rep.model_dtype)
    return feat, attrs * cutoff, cutoff


def message(inter, features, attrs, h, density, cutoff, index, count):
    conv = inter.rejector
    projection = inter.edge_info.mlp[-1]
    if features.is_cuda:
        out = bias_convolution(
            conv.eqx_tp,
            conv.reshape_in(features),
            attrs,
            h,
            projection.get_weight(),
            projection.bias,
            index,
            count,
        )
        out = conv.reshape_out(out)
    else:
        edge = conv.tp(features[index[0]], attrs, projection(h))
        out = edge.new_zeros(count, edge.shape[1]).index_add_(0, index[1], edge)
    norm = inter.scatter_norm
    denom = density.new_zeros(count, 1).index_add_(0, index[1], density * cutoff)
    return inter.linear_down(out) / (norm.alpha + norm.beta * denom)


def update_node(inter, product, message, previous, types, batch):
    m = inter.linear_nonlinearity(inter.nonlinearity(message))
    sc = None
    if previous is not None:
        if hasattr(inter, "stochastic_depth"):
            m = inter.stochastic_depth(m, batch)
        m = torch.add(inter.resnetBA(previous), m, alpha=inter.residual_scale)
        sc = inter.resnetAB(m)
        m = inter.norm2(m)
    return product(m, types, sc, batch)


class SevenRepresentation(NodeRepresentation):
    """Evaluate degree-three CG interactions with shared radial inputs."""

    def edge_geometry(self, vector, length, types, index):
        radial, attrs, cutoff = geometry(self, vector, length, types, index)
        if self.training and radial.is_cuda:
            heads = [
                radial_parameters(inter, edge)
                for inter, edge in zip(self.interactions, self.edge_updates)
            ]
            radial = project_radial_heads(
                radial, heads, types[index[0]], types[index[1]]
            )
        return radial, attrs, cutoff

    def edge_message(self, inter, edge, features, geometry, index, n, src, dst):
        feat, attrs, cutoff = geometry
        if isinstance(feat, tuple):
            h, density = radial_activations(inter, feat[inter.layer])
        else:
            h, density = radial_heads(
                inter, radial_parameters(inter, edge), feat, src, dst
            )
        return message(inter, features, attrs, h, density, cutoff, index, n)

    def forward(self, data, graph):
        types, index = graph.node_type, data["edge_index"]
        n = graph.lmp_natoms[0] if graph.lmp else types.shape[0]
        local = types[:n]
        src, dst = types[index[0]], types[index[1]]
        inputs = self.edge_geometry(graph.edge_vector, graph.edge_length, types, index)
        table = self.interactions[0].linear_up(self.node_embedding.embedding.weight)
        features = F.embedding(types, table)
        x = None
        descriptors = []
        for i, (inter, edge, product) in enumerate(
            zip(self.interactions, self.edge_updates, self.products)
        ):
            if i:
                features = inter.linear_up(inter.norm1(x))
                features = inter.handle_lammps(
                    features, graph.lmp_data, graph.lmp_natoms, i
                )
            msg = self.edge_message(inter, edge, features, inputs, index, n, src, dst)
            x = update_node(inter, product, msg, x, local, data["batch"])
            descriptors.append(x)
        return dict(
            descriptors=descriptors,
            uie_feats=None,
            noise_mask_tensor=None,
            dens_batch_mask_tensor=None,
            magnetic_radial_basis=None,
        )


def validate_representation(rep):
    """Validate the node representation for the shared radial-head schedule."""
    from tace.models.linear import mlpLinear

    if not (
        isinstance(rep, NodeRepresentation)
        and (rep.num_channel, rep.num_layers) == (128, 5)
        and len(rep.interactions) == len(rep.edge_updates) == len(rep.products) == 5
        and type(rep.edge_embedding).__name__ == "NonLinearEdgeEmbedding"
        and not rep.use_dens
        and not rep.use_magnetic_interaction
        and not hasattr(rep, "uie_embedding")
        and rep.uee_embeddings is None
        and not hasattr(rep, "final_norm")
        and not rep.radial_basis.apply_cutoff
        and not any(has_lora(module) for module in rep.modules())
    ):
        raise ValueError("Expected the released TACE-7M architecture")
    if not plain_element_lookups(rep):
        raise ValueError("TACE-7M requires plain node and endpoint embedding lookups.")
    require_native_messages(rep)
    for i, (inter, edge, product) in enumerate(
        zip(rep.interactions, rep.edge_updates, rep.products)
    ):
        if not (
            type(inter).__name__ == "O3CgtpInteraction"
            and type(inter.scatter_norm).__name__ == "DensityScatterNorm"
            and inter.edge_info.dims == [384, 64, 64, 1920 if i else 512]
            and type(product).__name__ == "BilinearMoEACE"
            and product.agnostic
            and product.correlation == 2
            and product.use_bilinear_gate
            and not hasattr(inter, "resnetBB")
            and all(
                hasattr(inter, k) == bool(i)
                for k in ("resnetBA", "resnetAB", "norm1", "norm2")
            )
            and inter.resnet_linear_type == "identity"
            and tuple((m, ir.l) for m, ir in inter.rejector.irreps_in2)
            == tuple((1, l) for l in range(4))
            and len(inter.edge_info.mlp) == 7
            and len(inter.scatter_norm.edge_density.mlp) == 4
        ):
            raise ValueError(f"Unsupported TACE-7M layer {i}")
        head = inter.edge_info.mlp
        density = inter.scatter_norm.edge_density.mlp
        if not (
            all(
                type(affine) is mlpLinear and affine.bias is not None
                for affine in (head[0], density[0], head[-1])
            )
            and head[0].out_dim == density[0].out_dim == 64
            and density[0].in_dim == head[0].in_dim == 384
            and edge.source_embedding.embedding_dim
            == edge.target_embedding.embedding_dim
            == 128
        ):
            raise ValueError(
                f"Layer {i} requires the biased 7M radial heads and 128-wide endpoint embeddings."
            )


def optimize_oam_7m(model):
    """Lower TACE-OAM-7M or TACE-OMat24-7M using the original parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded TACE property adapter.

    Examples
    --------
    >>> fast = optimize_oam_7m(model)  # doctest: +SKIP
    """
    rep = getattr(getattr(model, "readout_fn", None), "representation", None)
    validate_representation(rep)
    from equivariantx_foundation.registry import require_cuda

    require_cuda(model)
    if isinstance(rep, SevenRepresentation):
        return model
    result = shared_parameter_copy(model)
    result.readout_fn.representation.__class__ = SevenRepresentation
    return result
