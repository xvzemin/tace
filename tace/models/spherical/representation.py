################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, List, Union

import torch
from e3nn import o3

from equivariantx import co3, o2
from equivariantx.conv import EdgePairs, indexed_rows

from ...dataset.quantity import PROPERTIES
from ...utils.env import acceleration_enabled, get_tace_use_dens
from ..layer_norm import get_normalization_layer, residual_update_count
from ..linear import e3nnLinear
from ..radial import RadialBasis
from ..time_reversal import supports_time_reversal
from .edge import (
    EDGE_EMBEDDING,
    EDGE_UPDATE,
    DegreeEdgeEmbedding,
    Element2EdgeEmbedding,
    ElementEdgeEmbedding,
)
from .edge_inter import EDGE_INTERACTION
from .inter import (
    INTERACTION,
    O2CgtpInteraction,
    O3CgtpInteraction,
    O3IctdInteraction,
    UuO2Interaction,
    UvO2Interaction,
)
from .mtace import MagneticBasis, O2MagneticInteraction, O3MagneticWigner6jInteraction
from .node import (
    NODE_EMBEDDING,
    NODE_UPDATE,
    SphericalTensorNodeEmbedding,
    WignerTensorNodeEmbedding,
)
from .prod import PRODUCT, CgtpACE
from .scatter_norm import CutoffScatterNorm
from .tece_oam_rra import BilinearMoEACE, UvSO2Interaction
from .ue import UniversalEquivariantEmbedding, UniversalInvariantEmbedding


class NodeRepresentation(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        atomic_numbers: List[int],
        cutoff: float,
        avg_num_neighbors: float,
        magnetic_scale,
        mmax: int,
        Lmax: int,
        lmax: int,
        num_channel: int,
        target_irreps: o3.Irreps,
        node_embedding: Dict,
        edge_embedding: Dict,
        edge_update: Dict,
        node_update: Union[Dict, None],
        radial_basis: Dict,
        angular_basis: Union[Dict, None],
        atomic_basis: Dict,
        resnet: Dict,
        product_basis: Dict,
        invariant_property: List[str],
        equivariant_property: List[str],
        universal_embedding: Dict,
        layer_norm: Dict,
        dropout: Dict,
        parity: bool | None,
        path_irreps: o3.Irreps | None = None,
        mlp_type: str = "mlp_v1",
    ):
        super().__init__()
        self.model_dtype = torch.get_default_dtype()
        path_irreps = target_irreps if path_irreps is None else path_irreps

        self.num_elements = len(atomic_numbers)
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.invariant_property = list(invariant_property)
        self.equivariant_property = list(equivariant_property)
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.use_dens = get_tace_use_dens() == "1"

        # === radial basis ===
        self.radial_basis = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis["num_radial_basis"],
            cutoff_fn=radial_basis["cutoff_fn"],
            polynomial_cutoff=radial_basis["polynomial_cutoff"],
            radial_basis=radial_basis["radial_basis"],
            distance_transform=radial_basis["distance_transform"],
            trainable=radial_basis["trainable"],
            apply_cutoff=radial_basis["apply_cutoff"],
            gaussian_width=radial_basis["gaussian_width"],
            lmax=lmax,
            max_eigenvalue=radial_basis["max_eigenvalue"],
            use_interpolation=radial_basis["use_interpolation"],
        )
        num_radial_basis = self.radial_basis.num_basis
        widths = self.radial_basis.widths

        # === module classes ===
        node_embedding_cls = NODE_EMBEDDING[node_embedding["type"]]
        interaction_classes = [
            INTERACTION[interaction] for interaction in atomic_basis["type"]
        ]
        equivalent_cgtp = (O3CgtpInteraction, O2CgtpInteraction, O3IctdInteraction)
        if widths is not None and any(
            cls not in equivalent_cgtp for cls in interaction_classes
        ):
            raise ValueError(
                f"{radial_basis['radial_basis']} requires harmonic-degree paths: "
                "use cgtp, co3 or o2_cgtp."
            )
        weighted_interactions = (*equivalent_cgtp, UvO2Interaction, UuO2Interaction)
        self.indexed_element_layers = tuple(
            cls in equivalent_cgtp for cls in interaction_classes
        )
        self.first_indexed_layer = next(
            (i for i, indexed in enumerate(self.indexed_element_layers) if indexed),
            None,
        )
        self.reciprocal_radial = (
            edge_update["type"] == "identity"
            and all(cls in weighted_interactions for cls in interaction_classes)
            and edge_embedding["type"]
            in ("identity", "linear", "nonlinear", "element", "element2")
        )
        if edge_embedding["type"] in ("element", "element2") and any(
            cls not in weighted_interactions for cls in interaction_classes
        ):
            raise ValueError(
                "Element edge embedding requires weighted convolution paths in every layer."
            )
        uses_so2_interaction = any(
            issubclass(interaction_cls, UvSO2Interaction)
            for interaction_cls in interaction_classes
        )
        uses_o2_interaction = any(
            issubclass(interaction_cls, (UvO2Interaction, UuO2Interaction))
            for interaction_cls in interaction_classes
        )
        uses_o2_cgtp_interaction = any(
            issubclass(interaction_cls, O2CgtpInteraction)
            for interaction_cls in interaction_classes
        )
        self.use_o3_angular_basis = (
            issubclass(node_embedding_cls, SphericalTensorNodeEmbedding)
            and not node_embedding_cls.use_wigner
        ) or any(
            not issubclass(interaction_cls, (O2CgtpInteraction, O3IctdInteraction))
            and interaction_cls not in (UvO2Interaction, UuO2Interaction)
            for interaction_cls in interaction_classes
        )
        uses_co3_interaction = any(
            issubclass(cls, O3IctdInteraction) for cls in interaction_classes
        )
        self.use_so2 = uses_so2_interaction
        self.use_o2 = (
            issubclass(node_embedding_cls, WignerTensorNodeEmbedding)
            or uses_o2_interaction
            or uses_o2_cgtp_interaction
        )
        self.use_local_frame = (
            uses_so2_interaction
            or uses_o2_interaction
            or issubclass(node_embedding_cls, WignerTensorNodeEmbedding)
        )
        self._can_pack_wigner = not issubclass(
            node_embedding_cls, WignerTensorNodeEmbedding
        )
        self._can_skip_wigner = (
            self._can_pack_wigner
            and not uses_so2_interaction
            and not any(issubclass(cls, UvO2Interaction) for cls in interaction_classes)
        )
        uses_o2_magnetic_interaction = any(
            issubclass(interaction_cls, O2MagneticInteraction)
            for interaction_cls in interaction_classes
        )
        self.use_magnetic_wigner6j = any(
            issubclass(interaction_cls, O3MagneticWigner6jInteraction)
            for interaction_cls in interaction_classes
        )
        uses_magnetic_interaction = (
            uses_o2_magnetic_interaction or self.use_magnetic_wigner6j
        )
        if self.use_magnetic_wigner6j and not angular_basis["use_spin_orbit_coupling"]:
            raise ValueError("w6j_mag requires use_spin_orbit_coupling=True.")
        uses_time_odd_property = any(
            PROPERTIES[name].get("time_reversal", 1) == -1
            for name in self.invariant_property + self.equivariant_property
        )
        self.use_time_reversal = (
            supports_time_reversal()
            and not uses_so2_interaction
            and (uses_magnetic_interaction or uses_time_odd_property)
        )
        if self.use_time_reversal:
            time_odd_scalars = [
                name
                for name in self.invariant_property
                if PROPERTIES[name].get("time_reversal", 1) == -1
            ]
            self.invariant_property = [
                name for name in self.invariant_property if name not in time_odd_scalars
            ]
            self.equivariant_property.extend(time_odd_scalars)
        self.use_magnetic_interaction = uses_magnetic_interaction
        self.use_magnetic_edge_attrs = uses_o2_magnetic_interaction
        self.magnetic_node_irreps_out = None
        self.magnetic_edge_irreps_out = None
        if self.use_magnetic_interaction:
            self.magnetic_basis = MagneticBasis(
                magnetic_scale,
                num_mag_radial_basis=radial_basis["num_mag_radial_basis"],
                Lmax=angular_basis["magnetic_Lmax"],
                atomic_numbers=atomic_numbers,
                time_reversal=self.use_time_reversal,
                use_spin_orbit_coupling=angular_basis["use_spin_orbit_coupling"],
                parity=parity,
                path_irreps=path_irreps,
            )
            self.magnetic_node_irreps_out = self.magnetic_basis.magnetic_node_irreps_out
            self.magnetic_edge_irreps_out = self.magnetic_basis.magnetic_edge_irreps_out

        # === angular basis ===
        if self.use_so2 and Lmax != lmax:
            raise ValueError("Legacy SO2 interactions require Lmax == lmax.")
        if uses_o2_cgtp_interaction and self.use_so2 and mmax != lmax:
            raise ValueError("Mixing o2_cgtp with legacy so2 requires mmax == lmax.")
        if self.use_local_frame:
            wigner_lmax = max(Lmax, lmax)
            if uses_o2_magnetic_interaction:
                wigner_lmax = max(wigner_lmax, self.magnetic_edge_irreps_out.lmax)
            self.o2_angular_basis = o2.WignerD(
                wigner_lmax
                if uses_o2_cgtp_interaction
                else mmax
                if uses_so2_interaction or uses_o2_interaction
                else 0,
                wigner_lmax,
                backend="cuda"
                if acceleration_enabled("eqx", kernel="conv")
                else "torch",
            )
        else:
            self.o2_angular_basis = None
        self.o3_angular_basis = (
            o3.SphericalHarmonics(
                o3.Irreps.spherical_harmonics(lmax, p=-1),
                normalize=False,
                normalization="component",
            )
            if self.use_o3_angular_basis
            or issubclass(node_embedding_cls, WignerTensorNodeEmbedding)
            else None
        )
        self.co3_angular_basis = (
            co3.CartesianHarmonics(
                o3.Irreps.spherical_harmonics(lmax, p=-1),
                normalize=False,
                normalization="component",
                backend="cuda"
                if acceleration_enabled("eqx", kernel="conv")
                else "torch",
            )
            if uses_co3_interaction
            else None
        )
        for angular_basis in (
            self.o2_angular_basis,
            self.o3_angular_basis,
            self.co3_angular_basis,
        ):
            if angular_basis is not None:
                angular_basis.geometry_precision = True

        # === node/edge embedding ===
        self.node_embedding = node_embedding_cls(
            num_elements=self.num_elements,
            num_radial_basis=num_radial_basis,
            num_channel=num_channel,
            Lmax=Lmax,
            lmax=lmax,
            avg_num_neighbors=avg_num_neighbors,
            bias=False,
            radial_mlp=radial_basis["hidden"],
            mlp_type=mlp_type,
            radial_mlp_norm=radial_basis["mlp_norm"],
        )
        embed = EDGE_EMBEDDING[edge_embedding["type"]]
        options = dict(
            num_elements=self.num_elements, num_channel=num_channel, bias=False
        )
        if widths is not None and edge_embedding["type"] in ("linear", "nonlinear"):
            self.edge_embedding = DegreeEdgeEmbedding(embed, widths, **options)
            widths = self.edge_embedding.output_widths
        else:
            self.edge_embedding = embed(num_radial_basis=num_radial_basis, **options)

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
                mlp_type=mlp_type,
            )

        # === Edge Update ===
        self.edge_updates = torch.nn.ModuleList(
            [
                EDGE_UPDATE[edge_update["type"]](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    num_radial_basis=num_radial_basis,
                    edge_embedding_channel=self.edge_embedding.out_dim,
                    num_channel=num_channel,
                )
                for layer in range(num_layers)
            ]
        )
        self.magnetic_edge_updates = (
            torch.nn.ModuleList(
                [
                    EDGE_UPDATE["identity"](
                        layer=layer,
                        num_layers=num_layers,
                        num_elements=self.num_elements,
                        num_radial_basis=radial_basis["num_mag_radial_basis"],
                        edge_embedding_channel=num_channel,
                        num_channel=num_channel,
                    )
                    if issubclass(cls, O3MagneticWigner6jInteraction)
                    else torch.nn.Identity()
                    for layer, cls in enumerate(interaction_classes)
                ]
            )
            if self.use_magnetic_wigner6j
            else None
        )
        self.node_updates = (
            torch.nn.ModuleList(
                [
                    NODE_UPDATE[
                        "element"
                        if issubclass(cls, O3MagneticWigner6jInteraction)
                        else (node_update or {}).get("magnetic_type", "identity")
                    ](
                        num_elements=self.num_elements,
                        num_radial_basis=radial_basis["num_mag_radial_basis"],
                        num_channel=num_channel,
                        bias=radial_basis["bias"],
                    )
                    for cls in interaction_classes
                ]
            )
            if uses_magnetic_interaction
            else None
        )

        # === Interaction ===
        for_interactions = {
            "num_layers": num_layers,
            "num_elements": self.num_elements,
            "avg_num_neighbors": avg_num_neighbors,
            "mmax": mmax,
            "Lmax": Lmax,
            "lmax": lmax,
            "num_channel": num_channel,
            "target_irreps": target_irreps,
            "path_irreps": path_irreps,
            "num_radial_basis": num_radial_basis,
            "radial_widths": widths,
            "num_mag_radial_basis": radial_basis["num_mag_radial_basis"],
            "radial_mlp": radial_basis["hidden"],
            "radial_bias": radial_basis["bias"],
            "mlp_type": mlp_type,
            "radial_mlp_norm": radial_basis["mlp_norm"],
            "l1l2": atomic_basis["l1l2"],
            "scatter_norm": atomic_basis["scatter_norm"],
            "correlation": product_basis["correlation"],
            "resnet_type": resnet["type"],
            "resnet_linear_type": resnet["linear_type"],
            "use_first_resnet": resnet["use_first_resnet"],
            "pre_norm_type": layer_norm["pre_norm_type"],
            "use_first_pre_norm": layer_norm["use_first_pre_norm"],
            "parity": parity,
            "bias": atomic_basis["bias"],
            "stochastic_depth": dropout["stochastic_depth"],
            "num_head": atomic_basis["num_head"],
            "use_asymmetric_contraction": atomic_basis["use_asymmetric_contraction"],
            "gate_m0": atomic_basis["gate_m0"],
            "scalar_act": atomic_basis["scalar_act"],
            "tensor_act": atomic_basis["tensor_act"],
            "edge_ace_hidden": atomic_basis["edge_ace_hidden"],
            "magnetic_edge_irreps": self.magnetic_edge_irreps_out,
            "magnetic_node_irreps": self.magnetic_node_irreps_out,
        }

        self.interactions = torch.nn.ModuleList()
        if len(self.equivariant_property) > 0:
            self.uee_embeddings = torch.nn.ModuleList()
        else:
            self.uee_embeddings = None
        self.products = torch.nn.ModuleList()

        num_residual_updates = 0
        if layer_norm["pre_norm_type"] == "const_norm":
            num_residual_updates = residual_update_count(
                num_layers,
                resnet["type"],
                resnet["use_first_resnet"],
                layer_norm["use_first_pre_norm"],
            )
        scale_first = resnet["use_first_resnet"] and layer_norm["use_first_pre_norm"]
        for layer in range(num_layers):
            layer_updates = num_residual_updates if layer > 0 or scale_first else 0
            # === Interaction ===
            self.interactions.append(
                interaction_classes[layer](
                    **for_interactions,
                    layer=layer,
                    num_residual_updates=layer_updates,
                    edge_feats_channel=self.edge_updates[layer].out_dim,
                    magnetic_node_info_channel=(
                        self.node_updates[layer].out_dim
                        if self.node_updates is not None
                        else 0
                    ),
                    magnetic_edge_feats_channel=(
                        self.magnetic_edge_updates[layer].out_dim
                        if issubclass(
                            interaction_classes[layer], O3MagneticWigner6jInteraction
                        )
                        else 0
                    ),
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
                        time_reversal=self.use_time_reversal,
                        parity=parity,
                    ),
                )
                prod_irreps_in = self.uee_embeddings[layer].irreps_out
            else:
                prod_irreps_in = inter_irreps_out

            # === Product ===
            product_cls = PRODUCT[product_basis["type"][layer]]
            if product_cls is CgtpACE and (
                product_basis["nonlinear"] is not None
                or (product_basis["num_expert"] or 1) > 1
            ):
                product_cls = BilinearMoEACE
            self.products.append(
                product_cls(
                    layer=layer,
                    num_layers=num_layers,
                    num_residual_updates=layer_updates,
                    num_elements=self.num_elements,
                    Lmax=Lmax,
                    lmax=lmax,
                    num_channel=num_channel,
                    num_expert=product_basis["num_expert"],
                    num_channel_per_expert=product_basis["num_channel_per_expert"],
                    nonlinear=product_basis["nonlinear"],
                    scalar_act=product_basis["scalar_act"],
                    target_irreps=target_irreps,
                    path_irreps=path_irreps,
                    correlation=product_basis["correlation"],
                    l1l2=product_basis["l1l2"],
                    bias=product_basis["bias"],
                    stochastic_depth=dropout["stochastic_depth"],
                    parity=parity,
                    irreps_in=prod_irreps_in,
                    use_shared_expert=product_basis["use_shared_expert"],
                    agnostic=product_basis["agnostic"],
                    use_time_reversal=self.use_time_reversal,
                )
            )
            self.irreps_out = self.products[-1].irreps_out

        if isinstance(self.edge_embedding, ElementEdgeEmbedding):
            self.edge_embedding.build_paths(
                inter.rejector for inter in self.interactions
            )
        self.use_cutoff_scatter_norm = any(
            isinstance(inter.scatter_norm, CutoffScatterNorm)
            for inter in self.interactions
        )
        if layer_norm["final_norm_type"] is not None:
            self.final_norm = get_normalization_layer(
                layer_norm["final_norm_type"], self.irreps_out
            )

        if self.use_dens:
            self.irreps_forces_sh = o3.Irreps.spherical_harmonics(lmax=Lmax)
            self.forces_embedding = e3nnLinear(
                self.irreps_forces_sh,
                self.interactions[0].irreps_out,
                bias=True,
            )

    @property
    def irreps_outs(self):
        return [prod.irreps_out for prod in self.products]

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.model_dtype = next(self.node_embedding.parameters()).dtype
        return self

    @property
    def use_packed_wigner(self) -> bool:
        return (
            getattr(self, "use_local_frame", True)
            and getattr(self, "_can_pack_wigner", False)
            and all(
                interaction.use_eqx
                for interaction in self.interactions
                if isinstance(interaction, UvSO2Interaction)
            )
            and any(
                getattr(interaction, "use_eqx", False)
                for interaction in self.interactions
            )
        )

    def forward(self, data: Dict[str, torch.Tensor], graph) -> Dict[str, torch.Tensor]:

        node_type = graph.node_type
        node_type_local = node_type[: graph.lmp_natoms[0]] if graph.lmp else node_type

        # === edge initialize (radial) ===
        pairs = (
            EdgePairs(data["edge_pair_rows"], data["edge_pair_inverse"])
            if "edge_pair_rows" in data
            else None
        )
        radial_index = (
            data["edge_index"]
            if pairs is None
            else data["edge_index"][:, pairs.representatives]
        )
        radial_length = (
            graph.edge_length
            if pairs is None
            else graph.edge_length[pairs.representatives]
        )
        pair_basis, pair_cutoff = self.radial_basis(
            radial_length, node_type, radial_index, self.atomic_numbers
        )
        edge_cutoff = (
            pair_cutoff
            if pairs is None or pair_cutoff is None
            else pairs.expand(pair_cutoff)
        )
        message_cutoff = None
        if self.use_cutoff_scatter_norm:
            message_cutoff = CutoffScatterNorm.edge_weights(
                edge_cutoff, data["edge_index"][1], node_type.size(0)
            ).to(self.model_dtype)

        # === angular basis ===
        edge_wigner = None
        edge_wigner_inv = None
        skip_wigner = getattr(self, "_can_skip_wigner", False) and all(
            interaction.use_eqx
            for interaction in self.interactions
            if isinstance(interaction, UuO2Interaction)
        )
        if (
            getattr(self, "use_local_frame", self.use_so2 or self.use_o2)
            and not skip_wigner
        ):
            if getattr(self, "use_packed_wigner", False) and (
                not self.use_so2 or graph.edge_vector.is_cuda
            ):
                edge_wigner = self.o2_angular_basis.forward_packed(graph.edge_vector)
                edge_wigner_inv = edge_wigner
            else:
                edge_wigner, edge_wigner_inv = self.o2_angular_basis(graph.edge_vector)
        edge_attrs = (
            self.o3_angular_basis(
                graph.edge_vector / graph.edge_length.clamp_min(1e-12)
            )
            if getattr(self, "use_o3_angular_basis", True)
            else graph.edge_vector.new_empty((graph.edge_vector.size(0), 0))
        )
        cartesian_edge_attrs = (
            self.co3_angular_basis(
                graph.edge_vector / graph.edge_length.clamp_min(1e-12)
            )
            if getattr(self, "co3_angular_basis", None) is not None
            and not all(
                interaction.use_eqx
                for interaction in self.interactions
                if isinstance(interaction, O3IctdInteraction)
            )
            else None
        )

        pair_basis = pair_basis.to(self.model_dtype)
        edge_radial_basis = pair_basis if pairs is None else pairs.expand(pair_basis)
        edge_attrs = edge_attrs.to(self.model_dtype)
        if edge_cutoff is not None:
            edge_cutoff = edge_cutoff.to(self.model_dtype)
        if cartesian_edge_attrs is not None:
            cartesian_edge_attrs = cartesian_edge_attrs.to(self.model_dtype)
        if isinstance(edge_wigner, dict):
            edge_wigner = {
                key: value.to(self.model_dtype) for key, value in edge_wigner.items()
            }
            edge_wigner_inv = {
                key: value.to(self.model_dtype)
                for key, value in edge_wigner_inv.items()
            }
        elif edge_wigner is not None:
            edge_wigner = edge_wigner.to(self.model_dtype)
            edge_wigner_inv = edge_wigner_inv.to(self.model_dtype)

        initial_noncollinear_magmoms = data.get("initial_noncollinear_magmoms")
        magnetic_radial_basis = None
        magnetic_node_attrs = None
        magnetic_edge_attrs = None
        if self.use_magnetic_interaction:
            if initial_noncollinear_magmoms is None:
                raise ValueError(
                    "A magnetic model requires initial_noncollinear_magmoms"
                )
            (
                magnetic_radial_basis,
                magnetic_node_attrs,
                magnetic_edge_attrs,
            ) = self.magnetic_basis(
                initial_noncollinear_magmoms,
                node_type,
                data["edge_index"],
                node_fidelity=(
                    graph.node_fidelity[:1] if graph.lmp else graph.node_fidelity
                ),
                compute_edge=getattr(self, "use_magnetic_edge_attrs", True),
            )
            magnetic_radial_basis = magnetic_radial_basis.to(self.model_dtype)
            magnetic_node_attrs = magnetic_node_attrs.to(self.model_dtype)
            if magnetic_edge_attrs is not None:
                magnetic_edge_attrs = magnetic_edge_attrs.to(self.model_dtype)

        # === node initialize ===
        node_feats = self.node_embedding(
            node_type,
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
            node_type,
            pair_basis,
            radial_index,
            None if pair_cutoff is None else pair_cutoff.to(self.model_dtype),
        )
        directed_elements = isinstance(self.edge_embedding, Element2EdgeEmbedding)
        endpoint_types = (
            node_type[data["edge_index"] if directed_elements else radial_index]
            if isinstance(self.edge_embedding, ElementEdgeEmbedding)
            else None
        )
        indexed_types = None
        if (
            endpoint_types is not None
            and not directed_elements
            and self.first_indexed_layer is not None
            and self.interactions[self.first_indexed_layer].use_eqx
        ):
            indexed_types = indexed_rows(
                endpoint_types,
                self.num_elements,
                dtype=edge_feats.dtype,
                reduce="mean",
                backend=self.interactions[
                    self.first_indexed_layer
                ].rejector.eqx_tp.backend,
            )

        forces_embedding = None
        noise_mask_tensor = None
        dens_batch_mask_tensor = None
        if self.training and self.use_dens:
            forces_embedding, noise_mask_tensor, dens_batch_mask_tensor = (
                self._forward_dens_forces_encoding(data)
            )

        # === representation Learning ===
        prev_feats = []
        for idx, (edge_update, inter, prod) in enumerate(
            zip(self.edge_updates, self.interactions, self.products)
        ):
            node_types = node_type
            local_node_type = node_type
            this_edge_feats = edge_update(
                node_feats,
                node_types,
                edge_feats,
                data["edge_index"],
                edge_cutoff,
            )
            magnetic_node_info = (
                self.node_updates[idx](
                    magnetic_radial_basis,
                    node_types,
                )
                if self.node_updates is not None
                else None
            )
            if graph.lmp and idx > 0:
                local_node_type = node_type_local
            interaction_inputs = {}
            if pairs is not None:
                interaction_inputs["edge_pairs"] = pairs
            if isinstance(inter, O3MagneticWigner6jInteraction):
                interaction_inputs = {
                    "magnetic_node_attrs": magnetic_node_attrs,
                    "magnetic_edge_feats": self.magnetic_edge_updates[idx](
                        node_feats,
                        node_types,
                        magnetic_node_info[0][data["edge_index"][0]],
                        data["edge_index"],
                        edge_cutoff,
                    ),
                }
            if isinstance(self.edge_embedding, ElementEdgeEmbedding):
                interaction_inputs["element"] = (
                    self.edge_embedding.layer_weights(idx),
                    indexed_types
                    if indexed_types is not None and self.indexed_element_layers[idx]
                    else endpoint_types,
                )
            node_feats, sc = inter(
                node_feats,
                node_types,
                local_node_type,
                edge_radial_basis,
                this_edge_feats,
                cartesian_edge_attrs
                if isinstance(inter, O3IctdInteraction)
                else edge_attrs,
                data["edge_index"],
                edge_cutoff,
                edge_wigner,
                edge_wigner_inv,
                magnetic_node_info,
                magnetic_edge_attrs,
                data["batch"],
                graph,
                message_cutoff=message_cutoff,
                **interaction_inputs,
            )
            if graph.lmp and idx == 0:
                local_node_type = node_type_local
            if self.uee_embeddings is not None:
                node_feats = self.uee_embeddings[idx](node_feats, local_node_type, data)
            if forces_embedding is not None and idx == 0:
                node_feats = node_feats + forces_embedding
            node_feats = prod(
                node_feats,
                local_node_type,
                sc,
                data["batch"],
            )
            if idx == self.num_layers - 1 and hasattr(self, "final_norm"):
                node_feats = self.final_norm(node_feats)
            prev_feats.append(node_feats)

        return {
            "descriptors": prev_feats,
            # "edge_descriptors": edge_descriptors,
            "uie_feats": uie_feats,
            "noise_mask_tensor": noise_mask_tensor,
            "dens_batch_mask_tensor": dens_batch_mask_tensor,
            "magnetic_radial_basis": magnetic_radial_basis,
        }

    def _generate_dens_data(self, data: Dict[str, torch.Tensor]):
        num_atoms = data["node_type"].size(0)
        num_graphs = len(data["ptr"]) - 1
        device = data["node_type"].device

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
        forces_embedding = self.forces_embedding(forces_embedding.to(self.model_dtype))
        forces_embedding = forces_embedding * noise_mask
        return forces_embedding, noise_mask, dens_batch_mask

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for angular_basis in ("so2_angular_basis", "o2_angular_basis"):
            state_dict.pop(
                f"{prefix}{angular_basis}.wigner_index_to_m_array",
                None,
            )
            state_dict.pop(
                f"{prefix}{angular_basis}.wigner_inv_rescale",
                None,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PairRepresentation(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_sh: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        num_channel: int,
        atomic_numbers: list[int],
        cutoff: float,
        radial_basis: dict,
        interaction_type: str = "cgtp",
        mlp_type: str = "mlp_v1",
    ) -> None:
        super().__init__()
        self.num_layers = 1
        if cutoff <= 0:
            raise ValueError("Edge cutoff must be positive.")
        self.cutoff = float(cutoff)
        self.irreps_in = o3.Irreps(irreps_in)
        self.atomic_numbers = list(atomic_numbers)
        self.radial_basis = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis["num_radial_basis"],
            cutoff_fn=radial_basis["cutoff_fn"],
            polynomial_cutoff=radial_basis["polynomial_cutoff"],
            radial_basis=radial_basis["radial_basis"],
            distance_transform=radial_basis["distance_transform"],
            trainable=radial_basis["trainable"],
            apply_cutoff=False,
            gaussian_width=radial_basis["gaussian_width"],
            lmax=o3.Irreps(irreps_sh).lmax,
            max_eigenvalue=radial_basis["max_eigenvalue"],
            use_interpolation=radial_basis["use_interpolation"],
        )
        if self.radial_basis.widths is not None and interaction_type != "cgtp":
            raise ValueError(f"{radial_basis['radial_basis']} requires a cgtp pair interaction.")
        if interaction_type not in EDGE_INTERACTION:
            raise ValueError("interaction_type must be 'cgtp' or 'o2'.")
        self.interactions = torch.nn.ModuleList()
        edge_irreps = self.irreps_in
        for _ in range(self.num_layers):
            interaction = EDGE_INTERACTION[interaction_type](
                edge_irreps,
                irreps_sh,
                irreps_out,
                num_channel,
                len(atomic_numbers),
                dict(
                    radial_basis,
                    num_radial_basis=self.radial_basis.num_basis,
                    widths=self.radial_basis.widths,
                ),
                mlp_type=mlp_type,
            )
            self.interactions.append(interaction)
            edge_irreps = interaction.irreps_out
        self.irreps_out = edge_irreps
        self.o3_angular_basis = None
        self.o2_angular_basis = None
        if interaction_type == "cgtp":
            self.o3_angular_basis = o3.SphericalHarmonics(
                irreps_sh, normalize=True, normalization="component"
            )
        else:
            lmax = max(self.irreps_in.lmax, self.irreps_out.lmax)
            self.o2_angular_basis = o2.WignerD(
                lmax,
                lmax,
                method="auto"
                if acceleration_enabled("eqx", kernel="conv")
                else "recursive",
                backend="cuda"
                if acceleration_enabled("eqx", kernel="conv")
                else "torch",
            )
        for angular_basis in (self.o2_angular_basis, self.o3_angular_basis):
            if angular_basis is not None:
                angular_basis.geometry_precision = True

    def forward(
        self,
        node_feats: torch.Tensor,
        node_type: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shifts: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        model_dtype = node_feats.dtype
        positions = positions.to(torch.float64)
        lattice = lattice.to(torch.float64)
        source, target = edge_index
        edge_vector = (
            positions[target]
            - positions[source]
            + torch.einsum(
                "ei,eij->ej", edge_shifts.to(lattice.dtype), lattice[batch[source]]
            )
        )
        edge_length = torch.linalg.vector_norm(edge_vector, dim=-1, keepdim=True)
        atomic_numbers = torch.tensor(
            self.atomic_numbers, dtype=torch.int64, device=edge_vector.device
        )
        edge_radial_basis, edge_cutoff = self.radial_basis(
            edge_length, node_type, edge_index, atomic_numbers
        )
        edge_attrs = (
            self.o3_angular_basis(edge_vector)
            if self.o3_angular_basis is not None
            else None
        )
        edge_wigner = (
            self.o2_angular_basis.forward_packed(edge_vector)
            if self.o2_angular_basis is not None
            else None
        )
        edge_radial_basis = edge_radial_basis.to(model_dtype)
        if edge_attrs is not None:
            edge_attrs = edge_attrs.to(model_dtype)
        if edge_wigner is not None:
            edge_wigner = edge_wigner.to(model_dtype)
        return self.interactions[0](
            node_feats,
            edge_index,
            edge_radial_basis,
            edge_attrs,
            edge_cutoff.to(model_dtype),
            edge_wigner,
            node_type,
        )
