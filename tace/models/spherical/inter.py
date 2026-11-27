################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from equivariantx.conv import indexed_weights
from tace.utils.env import acceleration_enabled, memory_first

from ..lammps import Graph
from ..layer_norm import get_normalization_layer
from ..linear import IndexedFeatures, e3nnLinear, radial_features
from ..mlp import ACTIVATION, BATCH_MLP, MLP, convolution_radial
from .base import Interaction
from .fused import (
    O2TensorProductConv,
    O3CartesianTensorProductConv,
    O3TensorProductConv,
    UuO2TensorProductConv,
)
from .nonlinear import get_nonlinear_layer
from .o2 import UvO2TensorProductConv
from .residual import get_resnet_layer
from .scatter_norm import CutoffScatterNorm, get_scatter_norm_layer


class O3CgtpInteraction(Interaction):
    """Clebsch--Gordan convolution followed by node aggregation and updates."""

    @property
    def use_eqx(self) -> bool:
        return hasattr(self.rejector, "eqx_tp") and bool(
            acceleration_enabled("eqx", kernel="conv")
        )

    def _prepare_setup(self) -> None:
        pass

    def _build_rejector(self) -> torch.nn.Module:
        return O3TensorProductConv(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return self.rejector.irreps_out.simplify()

    def _setup_additional_modules(self) -> None:
        pass

    def _radial_projection(self, edge_feats, element=None, edge_pairs=None):
        if self.radial_widths is not None:
            radial, projection, radial_network = self.edge_info.stream(
                edge_feats,
                backend=self.rejector.eqx_tp.backend,
                paired=edge_pairs is not None,
            )
            bias = None
        else:
            radial, layers = convolution_radial(self.edge_info, edge_feats)
            last = layers[-1]
            projection, bias = last.get_weight(), last.bias
            radial_network = (
                layers[:-1] if len(layers) > 1 or isinstance(radial, tuple) else None
            )
        if element is not None:
            table, endpoints = element
            ordered = isinstance(table, tuple)
            if ordered:
                terms = (
                    (table[0].to(projection) * 0.5, endpoints[0]),
                    (table[1].to(projection) * 0.5, endpoints[1]),
                )
            else:
                terms = ((table.to(projection), endpoints),)
            return indexed_weights(
                radial,
                projection,
                terms,
                radial_network=radial_network,
                bias=bias,
                radial_mode="full" if memory_first() else "last",
                edge_pairs=edge_pairs if ordered else None,
            )
        if bias is not None:
            projection = torch.cat((projection, bias.unsqueeze(0)), dim=0)
            if radial_network is None:
                radial = torch.cat((radial, torch.ones_like(radial[:, :1])), dim=-1)
        return radial, projection, radial_network

    def _conv_weights(self, edge_feats, element, edge_pairs=None):
        weights = self.edge_info(edge_feats)
        if element is not None:
            table, endpoints = element
            if edge_pairs is not None and isinstance(table, tuple):
                weights = edge_pairs.expand(weights)
                edge_pairs = None
            a, b = table if isinstance(table, tuple) else (table, table)
            offset = a[endpoints[0]] + b[endpoints[1]]
            weights = torch.add(weights, offset.to(weights), alpha=0.5)
        return weights if edge_pairs is None else edge_pairs.expand(weights)

    def _setup(self) -> None:
        self._prepare_setup()

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )

        self.rejector = self._build_rejector()

        (
            self.nonlinearity,
            self.linear_nonlinearity,
            linear_down_irreps_out,
        ) = get_nonlinear_layer(
            self.nonlinear,
            o3.Irreps([(self.num_channel, ir) for _, ir in self.irreps_out]),
            self.irreps_out,
            gate_m0=self.gate_m0,
            bias=self.use_bias,
            scalar_act=self.scalar_act,
            tensor_act=self.tensor_act,
        )

        self.linear_down = e3nnLinear(
            self._linear_down_irreps_in(),
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        radial_norm = self.radial_mlp_norm
        if radial_norm is None:
            radial_norm = (
                "layer_norm"
                if self.mlp_type == "mlp_v1" and self.radial_layer_norm
                else None
            )

        options = dict(
            bias=self.radial_bias,
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
        )
        if self.radial_widths is None:
            self.edge_info = MLP[self.mlp_type](
                [self.edge_feats_channel]
                + self.radial_mlp
                + [self.rejector.weight_numel],
                **options,
            )
        else:
            self.edge_info = BATCH_MLP[self.mlp_type].from_product(
                self.rejector,
                self.radial_widths,
                self.radial_mlp,
                context_features=self.edge_feats_channel - sum(self.radial_widths),
                **options,
            )
        self._setup_additional_modules()

        self.scatter_norm = get_scatter_norm_layer(
            self.scatter_norm_type,
            avg_num_neighbors=self.avg_num_neighbors,
            edge_feats_channel=self.edge_feats_channel,
            radial_bias=self.radial_bias,
            radial_layer_norm=self.radial_layer_norm,
            mlp_type=self.mlp_type,
        )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BB":
            self.resnetBB = get_resnet_layer(
                self.irreps_in,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BAB":
            self.resnetBA = get_resnet_layer(
                self.irreps_in,
                self.irreps_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )
            if (
                self.layer > 0 or self.use_first_dropout
            ) and self.stochastic_depth_p > 0.0:
                from .dropout import GraphDropPath

                self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in [
            "AB",
            "BAB",
        ]:
            self.resnetAB = get_resnet_layer(
                self.irreps_out,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if self.pre_norm_type not in (None, "const_norm") and (
            self.use_first_pre_norm or self.layer > 0
        ):
            if self.resnet_type in ["BB", "BAB"]:
                self.norm1 = get_normalization_layer(self.pre_norm_type, self.irreps_in)
            if self.resnet_type in ["AB", "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type, self.irreps_out
                )

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_node_info: Union[tuple[torch.Tensor, torch.Tensor], None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
        graph: Union[Graph, None] = None,
        element=None,
        edge_pairs=None,
    ) -> torch.Tensor:
        # ML-IAP gathers from owned and ghost atoms, but pair_i only indexes
        # owned receivers. Other convolution providers keep their own contract.
        receiver = (
            {"num_nodes": graph.lmp_natoms[0]}
            if graph is not None
            and graph.lmp_data is not None
            and getattr(self.rejector, "supports_receiver_count", False)
            else {}
        )
        if self.use_eqx:
            radial, projection, radial_network = self._radial_projection(
                edge_feats, element, edge_pairs
            )
            # Mixed-precision geometry already produced the spherical harmonics
            # before the interaction. The native vector kernel requires like dtypes.
            reuse_attrs = (
                isinstance(self.rejector, O3TensorProductConv)
                and edge_attrs is not None
                and edge_attrs.shape[-1] == self.rejector.eqx_tp.edge_dim
                and graph is not None
                and graph.edge_vector.dtype != node_feats.dtype
            )
            return self.rejector.forward_stream(
                node_feats,
                edge_attrs,
                radial,
                projection,
                edge_index,
                edge_cutoff,
                edge_vector=graph.edge_vector / graph.edge_length.clamp_min(1e-12)
                if graph is not None and not reuse_attrs
                else None,
                radial_network=radial_network,
                edge_pairs=None
                if element is not None and isinstance(element[0], tuple)
                else edge_pairs,
                **receiver,
            )
        conv_weights = self._conv_weights(edge_feats, element, edge_pairs)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(
            node_feats, edge_attrs, conv_weights, edge_index, **receiver
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        local_node_type: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        magnetic_node_info: Union[tuple[torch.Tensor, torch.Tensor], None],
        magnetic_edge_attrs: Union[torch.Tensor, None],
        batch: torch.Tensor,
        graph: Graph,
        magnetic_node_attrs: torch.Tensor | None = None,
        magnetic_edge_feats: torch.Tensor | IndexedFeatures | None = None,
        message_cutoff: torch.Tensor | None = None,
        element: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        edge_pairs=None,
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None
        num_receivers = nlocal if nlocal is not None else node_types.size(0)

        # The reference last-layer path expects dense inputs; full streaming
        # retains the partitions until each edge tile is evaluated.
        if (
            isinstance(edge_feats, tuple)
            and self.radial_widths is None
            and self.edge_info.num_layers == 1
            and not self.use_eqx
        ):
            edge_feats = torch.cat(
                [
                    value if index is None else value[index]
                    for value, index in edge_feats
                ],
                dim=-1,
            )

        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, "resnetBB"):
            if self.resnet_linear_type == "aware":
                resBB = self.resnetBB(node_feats, local_node_type)
            else:
                resBB = self.resnetBB(node_feats)

        if hasattr(self, "resnetBA"):
            if self.resnet_linear_type == "aware":
                resBA = self.resnetBA(node_feats, local_node_type)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, "norm1"):
            node_feats = self.norm1(node_feats)

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        message_inputs = (
            {
                "magnetic_node_attrs": magnetic_node_attrs,
                "magnetic_edge_feats": magnetic_edge_feats,
            }
            if magnetic_node_attrs is not None
            else {}
        )
        if element is not None:
            message_inputs["element"] = element
        if isinstance(self.scatter_norm, CutoffScatterNorm):
            if message_cutoff is None:
                message_cutoff = self.scatter_norm.edge_weights(
                    edge_cutoff, edge_index[1], num_receivers
                )
        else:
            message_cutoff = edge_cutoff
        if edge_pairs is not None:
            message_inputs["edge_pairs"] = edge_pairs
        m_i = self.linear_down(
            self.truncate_ghosts(
                self._compute_messages(
                    node_feats,
                    node_types,
                    edge_radial_basis,
                    edge_feats,
                    edge_attrs,
                    edge_index,
                    message_cutoff,
                    edge_wigner,
                    edge_wigner_inv,
                    magnetic_node_info,
                    magnetic_edge_attrs,
                    graph=graph,
                    **message_inputs,
                ),
                nlocal,
            )
        )

        m_i = self.scatter_norm(
            m_i,
            edge_feats,
            edge_index,
            edge_cutoff,
            num_receivers,
            edge_pairs=edge_pairs,
        )

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        if resBA is not None:
            if hasattr(self, "stochastic_depth"):
                m_i = self.stochastic_depth(m_i, batch)
            m_i = torch.add(resBA, m_i, alpha=self.residual_scale)

        if hasattr(self, "resnetAB"):
            if self.resnet_linear_type == "aware":
                resAB = self.resnetAB(m_i, local_node_type)
            else:
                resAB = self.resnetAB(m_i)

        if hasattr(self, "norm2"):
            m_i = self.norm2(m_i)

        sc = resBB if resBB is not None else resAB
        return m_i, self.truncate_ghosts(sc, nlocal)


class O3IctdInteraction(O3CgtpInteraction):
    """Evaluate CGTP interactions with Cartesian ICTD contractions.

    Node inputs and outputs use spherical coordinates. Intermediate tensor
    products use Cartesian storage with identical paths and radial weights.
    Projection back to spherical coordinates follows node aggregation and
    channel compression. The product basis is unchanged.
    """

    def _build_rejector(self) -> torch.nn.Module:
        return O3CartesianTensorProductConv(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _setup_additional_modules(self) -> None:
        from equivariantx.conv.co3.linear import Linear

        self.linear_down.linear = Linear(
            self.linear_down.irreps_in,
            self.linear_down.irreps_out,
            self.rejector.eqx_tp.harmonic_output_index,
            internal_weights=False,
            shared_weights=True,
            backend="cuda" if self.use_eqx else "torch",
        )


class O2CgtpInteraction(O3CgtpInteraction):
    """Evaluate an O(3) CGTP through coordinate-free O(2) restriction.

    The paths, radial weights, normalization, and output layout are identical
    to :class:`O3CgtpInteraction`. All required local orders are retained.
    ``TACE_USE_EQX=1`` selects the fused convolution at execution time.
    """

    def _build_rejector(self) -> torch.nn.Module:
        return O2TensorProductConv(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_node_info: Union[tuple[torch.Tensor, torch.Tensor], None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
        graph: Union[Graph, None] = None,
        element=None,
        edge_pairs=None,
    ) -> torch.Tensor:
        if self.use_eqx:
            radial, projection, radial_network = self._radial_projection(
                edge_feats, element, edge_pairs
            )
            return self.rejector.forward_stream(
                node_feats,
                radial,
                projection,
                edge_index,
                edge_wigner,
                edge_cutoff,
                graph,
                radial_network=radial_network,
                edge_pairs=None
                if element is not None and isinstance(element[0], tuple)
                else edge_pairs,
            )
        conv_weights = self._conv_weights(edge_feats, element, edge_pairs)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(
            node_feats, conv_weights, edge_index, edge_wigner, edge_wigner_inv, graph
        )


class UvO2Interaction(O3CgtpInteraction):
    """Global O3 interaction evaluated through local O2 irreps.

    The interaction concatenates source and target node features and applies
    radial weighting and ``linear -> gate -> linear`` before inverse rotation
    and scatter.
    """

    convolution_cls = UvO2TensorProductConv

    def _build_rejector(self) -> torch.nn.Module:
        rejector = self.convolution_cls(
            self.irreps_in,
            self.irreps_out,
            num_channel=self.num_channel,
            mmax=self.mmax,
            even_scalar_act=ACTIVATION[
                (
                    self.scalar_act[0]
                    if isinstance(self.scalar_act, list)
                    else self.scalar_act or "silu"
                ).removeprefix("scaled_")
            ](),
            odd_scalar_act=ACTIVATION[
                (
                    self.scalar_act[1] if isinstance(self.scalar_act, list) else "tanh"
                ).removeprefix("scaled_")
            ](),
            tensor_act=ACTIVATION[
                (self.tensor_act or "sigmoid").removeprefix("scaled_")
            ](),
        )
        return rejector

    def _apply_rejector(
        self,
        node_feats: torch.Tensor,
        magnetic_node_info: Union[tuple[torch.Tensor, torch.Tensor], None],
        magnetic_edge_attrs: Union[torch.Tensor, None],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        radial_network=None,
        edge_pairs=None,
        *,
        num_nodes=None,
    ) -> torch.Tensor:
        return self.rejector(
            node_feats,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_cutoff=edge_cutoff,
            radial_network=radial_network,
            edge_pairs=edge_pairs,
            **({"num_nodes": num_nodes} if num_nodes is not None else {}),
        )

    def _prepare_setup(self) -> None:
        super()._prepare_setup()
        if self.edge_nonlinear is None:
            raise ValueError("o2 requires edge_nonlinear to be set.")
        if not 0 <= self.mmax <= max(self.Lmax, self.lmax):
            raise ValueError("o2 requires 0 <= mmax <= max(Lmax, lmax).")

        if isinstance(self.scalar_act, list):
            if len(self.scalar_act) != 2 or not all(
                isinstance(act, str) for act in self.scalar_act
            ):
                raise TypeError("O2 scalar_act entries must be strings.")
        elif self.scalar_act is not None and not isinstance(self.scalar_act, str):
            raise TypeError(
                "O2 scalar_act must be None, a string, or a list of two strings "
                "for 0e and 0o."
            )
        if self.tensor_act is not None and not isinstance(self.tensor_act, str):
            raise TypeError("O2 tensor_act must be None or a string for lm gates.")

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_node_info: Union[tuple[torch.Tensor, torch.Tensor], None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
        graph: Union[Graph, None] = None,
        edge_pairs=None,
        element=None,
    ) -> torch.Tensor:
        if element is not None:
            conv_weights = self._conv_weights(edge_feats, element, edge_pairs)
            radial_network, edge_pairs = None, None
        elif self.use_eqx and self.rejector.eqx_tp is not None and node_feats.is_cuda:
            conv_weights, radial_network = (
                radial_features(edge_feats),
                self.edge_info.mlp,
            )
        else:
            conv_weights, radial_network = self.edge_info(edge_feats), None
            if edge_pairs is not None:
                conv_weights = edge_pairs.expand(conv_weights)
                edge_pairs = None
        # Magnetic and custom interaction hooks have distinct call contracts.
        # Do not pass a new keyword unless the original hook and a receiver-
        # aware provider are both installed (Cartesian UV uses another map).
        receiver = (
            {"num_nodes": graph.lmp_natoms[0]}
            if getattr(graph, "lmp_data", None) is not None
            and type(self)._apply_rejector is UvO2Interaction._apply_rejector
            and "_apply_rejector" not in vars(self)
            and "supports_receiver_count" in type(self.rejector).__dict__
            and self.rejector.supports_receiver_count
            else {}
        )
        return self._apply_rejector(
            node_feats,
            magnetic_node_info,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
            edge_cutoff,
            radial_network=radial_network,
            edge_pairs=edge_pairs,
            **receiver,
        )


class UuO2Interaction(O3CgtpInteraction):
    """Global O(3) interaction with an externally weighted O(2) UuLinear.

    Only source node features enter the local frame. A single channelwise
    linear map uses one weight per local path and channel from the edge MLP,
    without an additional activation on these weights. No edge gate,
    channel-mixing linear, or radial rotary attention is applied. Node-level
    linear maps, inverse rotation, and scatter retain their usual behavior.
    With EQX convolution acceleration enabled, both frame rotations and all
    local paths are fused with gather/scatter and a bounded radial projection.
    """

    def _prepare_setup(self) -> None:
        if not 0 <= self.mmax <= max(self.Lmax, self.lmax):
            raise ValueError("uu_o2 requires 0 <= mmax <= max(Lmax, lmax).")

    def _build_rejector(self) -> torch.nn.Module:
        return UuO2TensorProductConv(
            self.irreps_in,
            self.irreps_out,
            num_channel=self.num_channel,
            mmax=self.mmax,
        )

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_node_info: Union[tuple[torch.Tensor, torch.Tensor], None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
        graph: Union[Graph, None] = None,
        edge_pairs=None,
        element=None,
    ) -> torch.Tensor:
        # Gather from the complete owner+ghost source table. Only the built-in
        # spherical map (or an explicitly opted-in provider) has a separate
        # receiver extent; Cartesian and third-party rejectors keep their ABI.
        receiver = (
            {"num_nodes": graph.lmp_natoms[0]}
            if getattr(graph, "lmp_data", None) is not None
            and "supports_receiver_count" in type(self.rejector).__dict__
            and self.rejector.supports_receiver_count
            else {}
        )
        if element is not None:
            weights = self._conv_weights(edge_feats, element, edge_pairs)
            if self.use_eqx:
                return self.rejector.forward_stream(
                    node_feats,
                    weights,
                    weights.new_empty((0, weights.shape[-1])),
                    edge_index,
                    edge_wigner,
                    edge_cutoff,
                    graph,
                    **receiver,
                )
            return self.rejector(
                node_feats,
                weights,
                edge_index,
                edge_wigner,
                edge_wigner_inv,
                edge_cutoff,
                **receiver,
            )
        if self.use_eqx:
            radial, projection, radial_network = self._radial_projection(edge_feats)
            return self.rejector.forward_stream(
                node_feats,
                radial,
                projection,
                edge_index,
                edge_wigner,
                edge_cutoff,
                graph,
                radial_network=radial_network,
                edge_pairs=edge_pairs,
                **receiver,
            )
        conv_weights = self.edge_info(edge_feats)
        return self.rejector(
            node_feats,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_cutoff,
            **receiver,
        )


from .mtace import O2MagneticInteraction, O3MagneticWigner6jInteraction
from .tece_oam_rra import UvSO2Interaction

INTERACTION: Dict[str, type[Interaction]] = {
    "cgtp": O3CgtpInteraction,
    "co3": O3IctdInteraction,
    "o2_cgtp": O2CgtpInteraction,
    "so2": UvSO2Interaction,
    "uv_o2": UvO2Interaction,
    "uu_o2": UuO2Interaction,
    "uv_o2_mag": O2MagneticInteraction,
    "w6j_mag": O3MagneticWigner6jInteraction,
}

interaction_aliases = {
    "normal": "cgtp",
    "spectral": "cgtp",
    "uv_so2": "so2",
    "attn": "so2",
}

INTERACTION.update(
    {alias: INTERACTION[target] for alias, target in interaction_aliases.items()}
)
