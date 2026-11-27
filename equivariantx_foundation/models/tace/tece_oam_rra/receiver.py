"""Alternating CGTP and rotary-attention updates on complete receiver neighborhoods."""

import torch

from ..receiver import TACEReceiver
from .model import RRAInteraction


class RRAReceiver(TACEReceiver):
    """Evaluate alternating CGTP and rotary-attention layers in receiver tiles."""

    receiver_label = "TECE-RRA"

    def __init__(self, model, tile, *, fold_products=False):
        self.model = model.readout_fn
        self.rep = self.model.representation
        self.tile = tile
        self.retain_messages = False
        self.replay_first_at = self.replay_prefix_at = None
        self.energy_domains = None
        self.first = self.rep.node_embedding.embedding.weight
        self.fold_products = fold_products
        self.product_elements = ()
        self.product_table = None
        self.fidelity = model.get_fidelity_idx()
        self.readout_widths = [0] * (len(self.rep.products) - 1) + [
            self.rep.irreps_outs[-1].dim
        ]

    def prepare_elements(self, types):
        """Cache folded products for the element IDs in the current structure."""
        if self.fold_products and types.is_cuda and types.numel():
            from .frozen_product import FrozenBilinearProduct

            active = types.unique()
            elements = tuple(active.tolist())
            if self.product_elements != elements:
                table = types.new_full((len(self.first),), -1)
                table[active] = torch.arange(len(elements), device=types.device)
                self.folded = [
                    FrozenBilinearProduct(product, elements=active)
                    for product in self.rep.products
                ]
                self.product_elements, self.product_table = elements, table

    def geometry(self, vec, types, idx, *, o3, o2):
        rep = self.rep
        length = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        basis, cutoff = rep.radial_basis(length, types, idx, rep.atomic_numbers)
        basis, cutoff = basis.to(self.first.dtype), cutoff.to(self.first.dtype)
        radial = rep.edge_embedding(None, types, basis, idx, cutoff)
        attrs = rotation = inverse = None
        if o3:
            attrs = rep.o3_angular_basis(vec / length.clamp_min(1e-12)).to(basis.dtype)
        if o2:
            if vec.is_cuda:
                rotation = inverse = rep.o2_angular_basis.forward_packed(vec).to(
                    basis.dtype
                )
            else:
                rotation, inverse = (
                    v.to(basis.dtype) for v in rep.o2_angular_basis(vec)
                )
        return basis, cutoff, radial, attrs, rotation, inverse

    def tile_geometry(self, tile, vec, local):
        return self.geometry(
            vec,
            tile[-2],
            tile[3],
            o3=bool(len(vec)) and not local,
            o2=bool(len(vec)) and local,
        )

    def message(self, i, tile, vec, x, projected, local):
        _, _, _, idx, _, types, batch = tile
        n = len(batch)
        rep = self.rep
        inter = rep.interactions[i]
        basis, cutoff, radial, attrs, rotation, inverse = self.tile_geometry(
            tile, vec, local
        )
        feats = rep.edge_updates[i](x, types, radial, idx, cutoff)
        if not len(vec):
            m = x.new_zeros(n, inter._linear_down_irreps_in().dim)
        else:
            m = inter._compute_messages(
                projected, types, basis, feats, attrs, idx, cutoff, rotation, inverse
            )[:n]
        return inter.scatter_norm(inter.linear_down(m), feats, idx, cutoff, n)

    def layer(self, i, tile, vec, xs, xc):
        types, batch = tile[-2:]
        n = len(batch)
        rep = self.rep
        inter = rep.interactions[i]
        x = self.first[types] if i == 0 else torch.cat((xc, xs))
        projected = inter.linear_up(inter.norm1(x) if hasattr(inter, "norm1") else x)
        m = self.message(i, tile, vec, x, projected, isinstance(inter, RRAInteraction))
        m = inter.linear_nonlinearity(inter.nonlinearity(m))
        if hasattr(inter, "resnetBA"):
            m = torch.add(inter.resnetBA(xc), m, alpha=inter.residual_scale)
        sc = inter.resnetAB(m) if hasattr(inter, "resnetAB") else None
        if hasattr(inter, "norm2"):
            m = inter.norm2(m)
        if self.product_table is None:
            value = rep.products[i](m, types[:n], sc, batch)
        else:
            value = self.folded[i](m, self.product_table[types[:n]], sc, batch)
        if i == len(rep.products) - 1 and hasattr(rep, "final_norm"):
            value = rep.final_norm(value)
        return value, None, None
