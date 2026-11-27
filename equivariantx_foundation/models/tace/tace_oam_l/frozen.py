"""Frozen radial-tangent representation for TACE-OAM-L."""

import torch

from equivariantx.utils.metadata import compact_metadata

from .angular import edge_attributes
from .element_grouped import grouping
from .model import _OAMLRepresentation
from .projection import project
from .radial import prepare_radial
from .tangent import embedding, message, stream_message


class RadialTangentRepresentation(_OAMLRepresentation):
    saved_edge_limit = 0
    stream_heads = False

    def forward(self, data, graph):
        types, index = graph.node_type, data["edge_index"]
        local = graph.lmp_natoms[0] if graph.lmp else None
        local_types = types[:local]
        groups = grouping(local_types, self.num_elements)
        src, dst = index
        st, tt = types[src].contiguous(), types[dst].contiguous()
        length = graph.edge_length
        if length.dtype != torch.float64:
            raise ValueError("Radial-tangent OAM-L requires FP64 edge geometry.")
        basis = self.radial_basis.radial_fn
        feats, fdot = embedding(
            length.detach(),
            basis.bessel_weights,
            basis.prefactor,
            self.frozen_embedding,
            self.edge_embedding.act1.acts[0].cst,
        )
        cutoff = self.radial_basis.cutoff_fn(length)
        attrs = edge_attributes(graph.edge_vector, length, cutoff, self.model_dtype)
        table = self.interactions[0].linear_up(self.node_embedding.embedding.weight)
        x = torch.nn.functional.embedding(types, table)
        descriptors = []
        for i, (inter, update, product) in enumerate(
            zip(self.interactions, self.edge_updates, self.products)
        ):
            residual = None
            if i:
                residual = inter.resnetBB(x, local_types, groups)
                x = inter.linear_up(x)
                x = inter.handle_lammps(x, graph.lmp_data, graph.lmp_natoms, i)
            p = prepare_radial(inter, update)
            conv = inter.rejector
            inp = conv.reshape_in(x)
            dummy = x.new_empty(1).expand(local_types.shape[0], conv.eqx_tp.output_dim)
            operands = (
                inp.contiguous(),
                attrs.contiguous(),
                length,
                feats,
                fdot,
                st,
                tt,
                src.contiguous(),
                dst.contiguous(),
                dummy,
                [q.contiguous() for q in p[:10]],
                p.info_eps,
                p.density_eps,
                compact_metadata(conv.eqx_tp.kernel_metadata),
            )
            if i and self.saved_edge_limit:
                z, d = project(*operands, self.saved_edge_limit)[:2]
            else:
                z, d = (stream_message if self.stream_heads else message)(*operands)
            norm = inter.scatter_norm
            total = d.new_zeros((local_types.shape[0], 1)).index_add(0, dst, d)
            x = inter.linear_down(conv.reshape_out(z)) / (
                total[:local] * norm.beta + norm.alpha
            )
            x = inter.linear_nonlinearity(inter.nonlinearity(x))
            x = product(x, local_types, residual, data["batch"], groups)
            descriptors.append(x)
        return {
            "descriptors": descriptors,
            "uie_feats": None,
            "noise_mask_tensor": None,
            "dens_batch_mask_tensor": None,
            "magnetic_radial_basis": None,
        }
