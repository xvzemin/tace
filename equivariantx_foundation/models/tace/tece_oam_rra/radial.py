"""Joint first projection of the eight trainable TECE radial networks."""

import torch
from torch.nn import functional as F

from equivariantx.utils.autograd import forward_ad_active
from equivariantx_foundation.models.tace.matching import plain_element_lookups
from equivariantx_foundation.models.tace.tace_oam_7m.model import (
    message,
    project_radial_heads,
    radial_activations,
    radial_parameters,
    update_node,
)
from tace.models.spherical.representation import NodeRepresentation


class RRARepresentation(NodeRepresentation):
    """Project the alternating CG and rotary radial heads jointly during training."""

    @staticmethod
    def supports(rep):
        if not plain_element_lookups(rep):
            return False
        return not rep.radial_basis.apply_cutoff and all(
            inter.edge_info.dims[:3] == [192, 64, 64]
            and len(inter.edge_info.mlp) == 7
            and inter.edge_info.mlp[0].bias is not None
            and all(
                hasattr(inter, key) == bool(i)
                for key in ("resnetAB", "resnetBA", "norm1", "norm2")
            )
            and (
                (i % 2 and type(inter.scatter_norm).__name__ == "IdentityScatterNorm")
                or (
                    type(inter.scatter_norm).__name__ == "DensityScatterNorm"
                    and inter.scatter_norm.edge_density.dims == [192, 64, 1]
                    and inter.scatter_norm.edge_density.mlp[0].bias is not None
                    and hasattr(inter.rejector, "eqx_tp")
                )
            )
            for i, inter in enumerate(rep.interactions)
        )

    def forward(self, data, graph):
        if (
            not self.training
            or self.model_dtype != torch.float32
            or not graph.edge_vector.is_cuda
            or graph.lmp
            or forward_ad_active()
            or "edge_pair_rows" in data
        ):
            return super().forward(data, graph)
        index, types = data["edge_index"], graph.node_type
        basis, cut = self.radial_basis(
            graph.edge_length, types, index, self.atomic_numbers
        )
        basis, cut = basis.to(self.model_dtype), cut.to(self.model_dtype)
        feat = self.edge_embedding(None, types, basis, index, cut)
        attrs = self.o3_angular_basis(
            graph.edge_vector / graph.edge_length.clamp_min(1e-12)
        ).to(self.model_dtype)
        matrix = self.o2_angular_basis.forward_packed(graph.edge_vector).to(
            self.model_dtype
        )
        radial = []
        for i, (inter, update) in enumerate(zip(self.interactions, self.edge_updates)):
            if i % 2:
                first = inter.edge_info.mlp[0]
                w, dst, src = first.get_weight().split(64)
                radial.append(
                    (
                        w,
                        update.source_embedding.weight @ src,
                        update.target_embedding.weight @ dst,
                        first.bias,
                    )
                )
            else:
                radial.append(radial_parameters(inter, update))
        heads = project_radial_heads(feat, radial, types[index[0]], types[index[1]])
        angular = attrs * cut
        features = F.embedding(
            types, self.interactions[0].linear_up(self.node_embedding.embedding.weight)
        )
        x = None
        descriptors = []
        for i, (inter, product, h) in enumerate(
            zip(self.interactions, self.products, heads)
        ):
            if i:
                features = inter.linear_up(inter.norm1(x))
            if i % 2:
                for j in range(1, len(inter.edge_info.mlp)):
                    h = inter.edge_info.mlp[j](h)
                msg = inter._compute_messages(
                    features,
                    types,
                    basis,
                    None,
                    attrs,
                    index,
                    cut,
                    matrix,
                    matrix,
                    graph=graph,
                    projected_weights=h,
                )
                msg = inter.linear_down(msg)
            else:
                hidden, den = radial_activations(inter, h)
                msg = message(
                    inter, features, angular, hidden, den, cut, index, len(types)
                )
            x = update_node(inter, product, msg, x, types, data["batch"])
            if i == len(self.interactions) - 1 and hasattr(self, "final_norm"):
                x = self.final_norm(x)
            descriptors.append(x)
        return dict(
            descriptors=descriptors,
            uie_feats=None,
            noise_mask_tensor=None,
            dens_batch_mask_tensor=None,
            magnetic_radial_basis=None,
        )
