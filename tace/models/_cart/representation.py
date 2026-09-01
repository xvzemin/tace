################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, List

import torch

from eqx import co3

from ...utils.env import get_tace_use_dens
from ..radial import RadialBasis
from .edge import EDGE_EMBEDDING, EDGE_UPDATE
from .inter import INTERACTION
from .layer_norm import get_normalization_layer
from .linear import Linear
from .mag import MagneticBasis
from .node import NODE_EMBEDDING
from .prod import PRODUCT
from .ue import UniversalEquivariantEmbedding, UniversalInvariantEmbedding


class Representation(torch.nn.Module):
    """Build Cartesian node descriptors from graph geometry and attributes."""

    def __init__(
        self,
        num_layers: int,
        atomic_numbers: List[int],
        cutoff: float,
        avg_num_neighbors: float,
        magmoms_norm_by_element,
        mmax: int,
        Lmax: int,
        lmax: int,
        mag_Lmax: int,
        num_channel: int,
        target_irreps,
        node_embedding: Dict,
        edge_embedding: Dict,
        edge_update: Dict,
        radial_basis: Dict,
        atomic_basis: Dict,
        resnet: Dict,
        product_basis: Dict,
        invariant_property: List[str],
        equivariant_property: List[str],
        universal_embedding: Dict,
        layer_norm: Dict,
        dropout: Dict,
        parity: bool,
        use_one_body_magmoms: bool,
    ) -> None:
        super().__init__()
        self.num_elements = len(atomic_numbers)
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.invariant_property = invariant_property
        self.equivariant_property = equivariant_property
        self.use_one_body_magmoms = use_one_body_magmoms
        self.use_magnetic_interaction = "o3_w6j_mag" in atomic_basis["type"]
        self.use_dens = get_tace_use_dens() == "1"
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )

        self.radial_basis = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis["num_radial_basis"],
            cutoff_fn=radial_basis["cutoff_fn"],
            polynomial_cutoff=radial_basis["polynomial_cutoff"],
            radial_basis=radial_basis["radial_basis"],
            distance_transform=radial_basis["distance_transform"],
            order=radial_basis["order"],
            trainable=radial_basis["trainable"],
            apply_cutoff=radial_basis["apply_cutoff"],
            gaussian_width=radial_basis["gaussian_width"],
        )
        self.angular_basis = co3.CartesianHarmonics(
            lmax,
            normalize=False,
            normalization="component",
        )
        self.magnetic_irreps = co3.CartesianHarmonics(
            mag_Lmax,
            irreps_in="1e",
            normalize=False,
            normalization="integral",
        ).irreps_out
        if self.use_one_body_magmoms or self.use_magnetic_interaction:
            self.magnetic_basis = MagneticBasis(
                magmoms_norm_by_element,
                num_basis=radial_basis["num_mag_radial_basis"],
                lmax=mag_Lmax,
                atomic_numbers=atomic_numbers,
            )

        self.node_embedding = NODE_EMBEDDING[node_embedding["type"]](
            num_elements=self.num_elements,
            num_radial_basis=self.radial_basis.num_basis,
            num_channel=num_channel,
            Lmax=Lmax,
            lmax=lmax,
            avg_num_neighbors=avg_num_neighbors,
            bias=False,
        )
        self.edge_embedding = EDGE_EMBEDDING[edge_embedding["type"]](
            num_elements=self.num_elements,
            num_radial_basis=self.radial_basis.num_basis,
            num_channel=num_channel,
            bias=False,
        )

        if self.invariant_property:
            self.uie_embedding = UniversalInvariantEmbedding(
                num_channel,
                {
                    name: config
                    for name, config in universal_embedding.items()
                    if name in self.invariant_property
                },
                bias=False,
            )

        self.edge_updates = torch.nn.ModuleList(
            EDGE_UPDATE[edge_update["type"]](
                layer=layer,
                num_layers=num_layers,
                num_elements=self.num_elements,
                num_radial_basis=self.radial_basis.num_basis,
                edge_embedding_channel=self.edge_embedding.out_dim,
                num_channel=num_channel,
            )
            for layer in range(num_layers)
        )

        interaction_kwargs = {
            "num_layers": num_layers,
            "num_elements": self.num_elements,
            "avg_num_neighbors": avg_num_neighbors,
            "mmax": mmax,
            "Lmax": Lmax,
            "lmax": lmax,
            "mag_Lmax": mag_Lmax,
            "num_channel": num_channel,
            "target_irreps": target_irreps,
            "num_radial_basis": radial_basis["num_radial_basis"],
            "num_mag_radial_basis": radial_basis["num_mag_radial_basis"] - 1,
            "radial_mlp": radial_basis["hidden"],
            "radial_bias": radial_basis["bias"],
            "l1l2": atomic_basis["l1l2"],
            "scatter_norm": atomic_basis["scatter_norm"],
            "correlation": product_basis["correlation"],
            "resnet_type": resnet["type"],
            "resnet_linear_type": resnet["linear_type"],
            "use_first_resnet": resnet["use_first_resnet"],
            "pre_norm_type": layer_norm["pre_norm_type"],
            "use_first_pre_norm": layer_norm["use_first_pre_norm"],
            "parity": parity,
            "bias": True,
            "stochastic_depth": dropout["stochastic_depth"],
            "num_head": atomic_basis["num_head"],
            "use_asymmetric_contraction": atomic_basis["use_asymmetric_contraction"],
            "use_radial_rotary_attention": atomic_basis["use_radial_rotary_attention"],
            "gate_m0": atomic_basis["gate_m0"],
            "scalar_act": atomic_basis["scalar_act"],
            "tensor_act": atomic_basis["tensor_act"],
            "edge_ace_hidden": atomic_basis["edge_ace_hidden"],
            "magnetic_irreps": self.magnetic_irreps,
        }

        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.uee_embeddings = (
            torch.nn.ModuleList() if self.equivariant_property else None
        )
        for layer in range(num_layers):
            interaction = INTERACTION[atomic_basis["type"][layer]](
                **interaction_kwargs,
                layer=layer,
                edge_feats_channel=self.edge_updates[layer].out_dim,
                nonlinear=atomic_basis["nonlinear"][layer],
                edge_nonlinear=atomic_basis["edge_nonlinear"][layer],
                irreps_in=(
                    self.node_embedding.irreps_out
                    if layer == 0
                    else self.products[layer - 1].irreps_out
                ),
            )
            self.interactions.append(interaction)
            product_irreps_in = interaction.irreps_out
            if self.uee_embeddings is not None:
                embedding = UniversalEquivariantEmbedding(
                    irreps_in=product_irreps_in,
                    num_channel=num_channel,
                    num_elements=self.num_elements,
                    config={
                        name: config
                        for name, config in universal_embedding.items()
                        if name in self.equivariant_property
                    },
                )
                self.uee_embeddings.append(embedding)
                product_irreps_in = embedding.irreps_out
            product = PRODUCT[product_basis["type"][layer]](
                layer=layer,
                num_layers=num_layers,
                num_elements=self.num_elements,
                Lmax=Lmax,
                lmax=lmax,
                num_channel=num_channel,
                num_expert=product_basis["num_expert"],
                num_channel_per_expert=product_basis["num_channel_per_expert"],
                nonlinear=product_basis["nonlinear"],
                scalar_act=product_basis["scalar_act"],
                target_irreps=target_irreps,
                correlation=product_basis["correlation"],
                l1l2=product_basis["l1l2"],
                bias=True,
                stochastic_depth=dropout["stochastic_depth"],
                parity=parity,
                irreps_in=product_irreps_in,
                use_shared_expert=product_basis["use_shared_expert"],
                agnostic=product_basis["agnostic"],
            )
            self.products.append(product)
            self.irreps_out = product.irreps_out

        if layer_norm["final_norm_type"] is not None:
            self.final_norm = get_normalization_layer(
                layer_norm["final_norm_type"], self.irreps_out
            )

        if self.use_dens:
            self.forces_harmonics = co3.CartesianHarmonics(
                Lmax,
                normalize=True,
                normalization="component",
            )
            self.forces_embedding = Linear(
                self.forces_harmonics.irreps_out,
                self.interactions[0].irreps_out,
                bias=True,
            )

    def forward(self, data: Dict[str, torch.Tensor], graph) -> Dict[str, object]:
        edge_radial_basis, edge_cutoff = self.radial_basis(
            graph.edge_length,
            data["node_attrs"],
            data["edge_index"],
            self.atomic_numbers,
        )
        edge_attrs = self.angular_basis(graph.edge_vector / graph.edge_length)
        node_feats = self.node_embedding(
            data["node_attrs"],
            edge_radial_basis,
            data["edge_index"],
            edge_attrs,
            edge_cutoff,
        )
        if hasattr(self, "uie_embedding"):
            uie_feats = self.uie_embedding(data)
            node_feats = node_feats + uie_feats
        else:
            uie_feats = None
        edge_feats = self.edge_embedding(
            node_feats,
            data["node_attrs"],
            edge_radial_basis,
            data["edge_index"],
            edge_cutoff,
        )

        forces_embedding = None
        noise_mask_tensor = None
        dens_batch_mask_tensor = None
        if self.training and self.use_dens:
            forces_embedding, noise_mask_tensor, dens_batch_mask_tensor = (
                self._forward_dens_forces_encoding(data)
            )

        one_body_magmoms_basis = None
        magnetic_radial_basis = None
        magnetic_node_attrs = None
        if self.use_one_body_magmoms or self.use_magnetic_interaction:
            magnetic_moments = data.get("initial_noncollinear_magmoms")
            if magnetic_moments is None:
                raise ValueError(
                    "initial_noncollinear_magmoms is required by the magnetic "
                    "one-body readout."
                )
            magnetic_basis, magnetic_node_attrs = self.magnetic_basis(
                magnetic_moments,
                data["node_attrs"],
            )
            if self.use_one_body_magmoms:
                one_body_magmoms_basis = magnetic_basis
            if self.use_magnetic_interaction:
                magnetic_radial_basis = magnetic_basis[..., 1:]

        descriptors = []
        for layer, (edge_update, interaction, product) in enumerate(
            zip(self.edge_updates, self.interactions, self.products)
        ):
            node_attrs = data["node_attrs"]
            this_edge_feats = edge_update(
                node_feats,
                data["node_attrs"],
                edge_feats,
                data["edge_index"],
                edge_cutoff,
            )
            if graph.lmp and layer > 0:
                node_attrs = node_attrs[: graph.lmp_natoms[0]]
            node_feats, sc = interaction(
                node_feats,
                data["node_attrs"],
                node_attrs,
                edge_radial_basis,
                this_edge_feats,
                edge_attrs,
                data["edge_index"],
                edge_cutoff,
                magnetic_radial_basis=magnetic_radial_basis,
                magnetic_node_attrs=magnetic_node_attrs,
                batch=data["batch"],
                graph=graph,
            )
            if graph.lmp and layer == 0:
                node_attrs = node_attrs[: graph.lmp_natoms[0]]
            if self.uee_embeddings is not None:
                node_feats = self.uee_embeddings[layer](node_feats, node_attrs, data)
            if forces_embedding is not None and layer == 0:
                node_feats = node_feats + forces_embedding
            node_feats = product(node_feats, node_attrs, sc, data["batch"])
            if layer == self.num_layers - 1 and hasattr(self, "final_norm"):
                node_feats = self.final_norm(node_feats)
            descriptors.append(node_feats)

        return {
            "descriptors": descriptors,
            "uie_feats": uie_feats,
            "noise_mask_tensor": noise_mask_tensor,
            "dens_batch_mask_tensor": dens_batch_mask_tensor,
            "one_body_magmoms_basis": one_body_magmoms_basis,
            "decouple_node_feats1": None,
            "decouple_node_feats2": None,
        }

    def _generate_dens_data(self, data: Dict[str, torch.Tensor]):
        num_atoms = data["node_attrs"].size(0)
        num_graphs = len(data["ptr"]) - 1
        device = data["node_attrs"].device
        forces_data = (
            data["direct_forces"] if "direct_forces" in data else data["forces"]
        )
        noise_mask = data.get(
            "noise_mask",
            torch.ones(num_atoms, dtype=torch.bool, device=device),
        ).view(-1, 1)
        dens_batch_mask = data.get(
            "dens_batch_mask",
            torch.ones(num_graphs, dtype=torch.bool, device=device),
        ).view(-1, 1)
        return forces_data, noise_mask, dens_batch_mask

    def _forward_dens_forces_encoding(self, data):
        forces, noise_mask, dens_batch_mask = self._generate_dens_data(data)
        forces_norm = forces.norm(dim=-1, keepdim=True) / math.sqrt(3.0)
        embedding = self.forces_embedding(self.forces_harmonics(forces))
        return embedding * forces_norm * noise_mask, noise_mask, dens_batch_mask


__all__ = ["Representation"]
