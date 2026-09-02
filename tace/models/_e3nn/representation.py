################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import math
from typing import Dict, List

import torch
from e3nn import o3

from eqx.o2 import WignerD

from ...utils.env import get_tace_use_dens
from ..angular import SphericalHarmonics
from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mag import MagneticBasis
from ..radial import RadialBasis
from .edge import EDGE_EMBEDDING, EDGE_UPDATE
from .inter import INTERACTION
from .layer_norm import get_normalization_layer
from .node import NODE_EMBEDDING
from .prod import PRODUCT
from .ue import UniversalEquivariantEmbedding, UniversalInvariantEmbedding


class Representation(torch.nn.Module):
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
        target_irreps: o3.Irreps,
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
    ):
        super().__init__()

        self.num_elements = len(atomic_numbers)
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.invariant_property = invariant_property
        self.equivariant_property = equivariant_property
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.resnet_type = resnet["type"]
        self.use_dens = get_tace_use_dens() == "1"

        # === radial basis ===
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

        # === angular basis ===
        self.use_o2 = any(
            interaction in ("o2", "o2_mag") for interaction in atomic_basis["type"]
        )
        self.use_legacy_so2 = (
            any(t.endswith("so2") for t in atomic_basis["type"])
            or any(t.endswith("attn") for t in atomic_basis["type"])
            or node_embedding["type"] == "so2_tensor"
        )
        self.use_so2 = self.use_legacy_so2 or self.use_o2
        self.use_o3 = (
            any(t != "so2" for t in atomic_basis["type"])
            or node_embedding["type"] == "tensor"
        )
        self.use_magnetic_radial_basis = any(
            interaction in {"o3_w6j_mag", "o2_mag"}
            for interaction in atomic_basis["type"]
        )
        self.use_one_body_magmoms = use_one_body_magmoms
        self.use_magnetic_node_attrs = any(
            interaction in {"o3_w6j_mag", "o2_mag"}
            for interaction in atomic_basis["type"]
        )
        self.magnetic_irreps = o3.Irreps.spherical_harmonics(mag_Lmax, p=1)
        if self.use_magnetic_radial_basis or self.use_one_body_magmoms:
            self.magnetic_basis = MagneticBasis(
                magmoms_norm_by_element,
                num_basis=radial_basis["num_mag_radial_basis"],
                magnetic_irreps=(
                    self.magnetic_irreps if self.use_magnetic_node_attrs else None
                ),
                atomic_numbers=atomic_numbers,
                num_elements=self.num_elements,
            )
        if self.use_so2:
            if self.use_legacy_so2 and Lmax != lmax:
                raise ValueError("Legacy SO2 interactions require Lmax == lmax.")
            self.so2_angular_basis = WignerD(mmax, max(Lmax, lmax))
        else:
            self.so2_angular_basis = None
        if self.use_o3:
            self.o3_angular_basis = SphericalHarmonics(
                o3.Irreps.spherical_harmonics(lmax, p=-1),
                normalize=False,
                normalization="component",
            )

        # === node/edge embedding ===
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

        # === universal embedding ===
        if len(self.invariant_property) > 0:
            self.uie_embedding = UniversalInvariantEmbedding(
                num_channel,
                {
                    k: v
                    for k, v in universal_embedding.items()
                    if k in self.invariant_property
                },
                bias=False,
            )

        # === Edge Update ===
        self.edge_updates = torch.nn.ModuleList(
            [
                EDGE_UPDATE[edge_update["type"]](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    num_radial_basis=self.radial_basis.num_basis,
                    edge_embedding_channel=self.edge_embedding.out_dim,
                    num_channel=num_channel,
                )
                for layer in range(num_layers)
            ]
        )

        # === Interaction ===
        for_interactions = {
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
        if len(self.equivariant_property) > 0:
            self.uee_embeddings = torch.nn.ModuleList()
        else:
            self.uee_embeddings = None
        self.products = torch.nn.ModuleList()

        for layer in range(num_layers):
            # === Interaction ===
            self.interactions.append(
                INTERACTION[atomic_basis["type"][layer]](
                    **for_interactions,
                    layer=layer,
                    edge_feats_channel=self.edge_updates[layer].out_dim,
                    nonlinear=atomic_basis["nonlinear"][layer],
                    edge_nonlinear=atomic_basis["edge_nonlinear"][layer],
                    irreps_in=self.node_embedding.irreps_out
                    if layer == 0
                    else self.products[layer - 1].irreps_out,
                )
            )
            inter_irreps_out = self.interactions[layer].irreps_out

            # === UEE ===
            if self.uee_embeddings is not None:
                self.uee_embeddings.append(
                    UniversalEquivariantEmbedding(
                        irreps_in=inter_irreps_out,
                        num_channel=num_channel,
                        num_elements=len(atomic_numbers),
                        config={
                            k: v
                            for k, v in universal_embedding.items()
                            if k in self.equivariant_property
                        },
                    ),
                )
                prod_irreps_in = self.uee_embeddings[layer].irreps_out
            else:
                prod_irreps_in = inter_irreps_out

            # === Product ===
            self.products.append(
                PRODUCT[product_basis["type"][layer]](
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
                    irreps_in=prod_irreps_in,
                    use_shared_expert=product_basis["use_shared_expert"],
                    agnostic=product_basis["agnostic"],
                )
            )
            self.irreps_out = self.products[-1].irreps_out

        if layer_norm["final_norm_type"] is not None:
            self.final_norm = get_normalization_layer(
                layer_norm["final_norm_type"],
                ls=target_irreps.ls,
                num_channels=num_channel,
            )
            self.final_reshape = LayoutTransform(
                [(num_channel, ir) for _, ir in target_irreps]
            )

        if self.use_dens:
            self.irreps_forces_sh = o3.Irreps.spherical_harmonics(lmax=Lmax)
            self.forces_embedding = e3nnLinear(
                self.irreps_forces_sh,
                self.interactions[0].irreps_out,
                bias=True,
            )

    def forward(self, data: Dict[str, torch.Tensor], graph) -> Dict[str, torch.Tensor]:

        # === edge initialize (radial) ===
        edge_radial_basis, edge_cutoff = self.radial_basis(
            graph.edge_length,
            data["node_attrs"],
            data["edge_index"],
            self.atomic_numbers,
        )

        # === angular basis ===
        edge_attrs = None
        edge_wigner = None
        edge_wigner_inv = None
        if self.use_so2:
            edge_wigner, edge_wigner_inv = self.so2_angular_basis.get_wigner(
                graph.edge_vector
            )
        if self.use_o3:
            edge_attrs = self.o3_angular_basis(
                graph.edge_vector / graph.edge_length
            )  # have added eps in adapter.py

        # === node initialize ===
        node_feats = self.node_embedding(
            data["node_attrs"],
            edge_radial_basis,
            data["edge_index"],
            edge_attrs,
            edge_cutoff,
            edge_wigner,
            edge_wigner_inv,
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

        initial_noncollinear_magmoms = data.get("initial_noncollinear_magmoms")
        magnetic_radial_basis = None
        magnetic_node_attrs = None
        one_body_magmoms_basis = None
        if self.use_magnetic_radial_basis or self.use_one_body_magmoms:
            if initial_noncollinear_magmoms is None:
                raise ValueError(
                    "A magnetic model requires initial_noncollinear_magmoms"
                )
            magnetic_basis, magnetic_node_attrs = self.magnetic_basis(
                initial_noncollinear_magmoms,
                data["node_attrs"],
            )
        if self.use_magnetic_radial_basis:
            magnetic_radial_basis = magnetic_basis[..., 1:]
        if self.use_one_body_magmoms:
            one_body_magmoms_basis = magnetic_basis
        # === representation Learning ===
        prev_feats = []
        for idx, (edge_update, inter, prod) in enumerate(
            zip(self.edge_updates, self.interactions, self.products)
        ):
            node_attrs_total = data["node_attrs"]
            node_attrs_slice = data["node_attrs"]
            this_edge_feats = edge_update(
                node_feats,
                node_attrs_total,
                edge_feats,
                data["edge_index"],
                edge_cutoff,
            )
            if graph.lmp and idx > 0:
                node_attrs_slice = node_attrs_slice[: graph.lmp_natoms[0]]
            node_feats, sc = inter(
                node_feats,
                node_attrs_total,
                node_attrs_slice,
                edge_radial_basis,
                this_edge_feats,
                edge_attrs,
                data["edge_index"],
                edge_cutoff,
                edge_wigner,
                edge_wigner_inv,
                magnetic_radial_basis,
                magnetic_node_attrs,
                data["batch"],
                graph,
            )
            if graph.lmp and idx == 0:
                node_attrs_slice = node_attrs_slice[: graph.lmp_natoms[0]]
            if self.uee_embeddings is not None:
                node_feats = self.uee_embeddings[idx](
                    node_feats, node_attrs_slice, data
                )
            if forces_embedding is not None and idx == 0:
                node_feats = node_feats + forces_embedding
            node_feats = prod(node_feats, node_attrs_slice, sc, data["batch"])
            if idx == self.num_layers - 1 and hasattr(self, "final_norm"):
                node_feats = self.final_reshape.inverse(
                    self.final_norm(self.final_reshape(node_feats))
                )
            prev_feats.append(node_feats)

        return {
            "descriptors": prev_feats,
            # "edge_descriptors": edge_descriptors,
            "uie_feats": uie_feats,
            "noise_mask_tensor": noise_mask_tensor,
            "dens_batch_mask_tensor": dens_batch_mask_tensor,
            "one_body_magmoms_basis": one_body_magmoms_basis,
        }

    def _generate_dens_data(self, data: Dict[str, torch.Tensor]):
        num_atoms = data["node_attrs"].size(0)
        num_graphs = len(data["ptr"]) - 1
        # dtype = data['node_attrs'].dtype
        device = data["node_attrs"].device

        if "direct_forces" in data:
            forces_data = data["direct_forces"]
        else:
            forces_data = data["forces"]
        if "noise_mask" in data:
            noise_mask = data["noise_mask"].view(-1, 1)
        else:
            noise_mask = torch.ones((num_atoms, 1), dtype=torch.bool, device=device)
        if "dens_batch_mask" in data:
            dens_batch_mask = data["dens_batch_mask"].view(-1, 1)
        else:
            dens_batch_mask = torch.ones(
                (num_graphs, 1), dtype=torch.bool, device=device
            )

        forces_sh = o3.spherical_harmonics(
            l=self.irreps_forces_sh,
            x=forces_data,
            normalize=True,
            normalization="component",
        )

        return forces_data, forces_sh, noise_mask, dens_batch_mask

    def _forward_dens_forces_encoding(self, data):
        forces_data, forces_sh, noise_mask, dens_batch_mask = self._generate_dens_data(
            data
        )
        forces_norm = forces_data.norm(dim=-1, keepdim=True)
        forces_norm = forces_norm / math.sqrt(3.0)
        forces_embedding = forces_sh * forces_norm  # [node, 3]
        forces_embedding = self.forces_embedding(forces_embedding)
        forces_embedding = forces_embedding * noise_mask
        return forces_embedding, noise_mask, dens_batch_mask
