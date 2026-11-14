################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from tace.utils.env import acceleration_enabled

from ..lammps import Graph
from ..layout import LayoutTransform
from ..linear import IndexedFeatures, e3nnLinear, radial_features
from ..mlp import ACTIVATION, MLP, convolution_radial
from .base import Interaction
from .fused import (
    O2TensorProductConv,
    O3CartesianTensorProductConv,
    O3TensorProductConv,
    UuO2TensorProductConv,
)
from .layer_norm import get_normalization_layer
from .nonlinear import get_nonlinear_layer
from .o2 import UvO2TensorProductConv
from .residual import get_resnet_layer
from .scatter_norm import get_scatter_norm_layer


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

    def _radial_projection(self, edge_feats):
        radial, layers = convolution_radial(self.edge_info, edge_feats)
        last = layers[-1]
        projection = last.get_weight()
        radial_network = (
            layers[:-1] if len(layers) > 1 or isinstance(radial, tuple) else None
        )
        if last.bias is not None:
            projection = torch.cat((projection, last.bias.unsqueeze(0)), dim=0)
            if radial_network is None:
                radial = torch.cat((radial, torch.ones_like(radial[:, :1])), dim=-1)
        return radial, projection, radial_network

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
        self.edge_info = MLP[self.mlp_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
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

        if (
            self.use_first_pre_norm or self.layer > 0
        ) and self.pre_norm_type is not None:
            if self.resnet_type in ["BB", "BAB"]:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.ls,
                    num_channels=self.num_channel,
                    irreps=self.irreps_in,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ["AB", "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.ls,
                    num_channels=self.num_channel,
                    irreps=self.irreps_out,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)

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
    ) -> torch.Tensor:
        if self.use_eqx:
            radial, projection, radial_network = self._radial_projection(edge_feats)
            return self.rejector.forward_stream(
                node_feats,
                edge_attrs,
                radial,
                projection,
                edge_index,
                edge_cutoff,
                edge_vector=graph.edge_vector / graph.edge_length.clamp_min(1e-12)
                if graph is not None
                else None,
                radial_network=radial_network,
            )
        conv_weights = self.edge_info(edge_feats)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(node_feats, edge_attrs, conv_weights, edge_index)

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
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        # The reference last-layer path expects dense inputs; full streaming
        # retains the partitions until each edge tile is evaluated.
        if (
            isinstance(edge_feats, tuple)
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
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        magnetic_inputs = (
            {
                "magnetic_node_attrs": magnetic_node_attrs,
                "magnetic_edge_feats": magnetic_edge_feats,
            }
            if magnetic_node_attrs is not None
            else {}
        )
        m_i = self.linear_down(
            self.truncate_ghosts(
                self._compute_messages(
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
                    graph=graph,
                    **magnetic_inputs,
                ),
                nlocal,
            )
        )

        m_i = self.scatter_norm(
            m_i, edge_feats, edge_index, edge_cutoff, node_types.size(0)
        )

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        if resBA is not None:
            if hasattr(self, "stochastic_depth"):
                m_i = self.stochastic_depth(m_i, batch)
            m_i = m_i + resBA

        if hasattr(self, "resnetAB"):
            if self.resnet_linear_type == "aware":
                resAB = self.resnetAB(m_i, local_node_type)
            else:
                resAB = self.resnetAB(m_i)

        if hasattr(self, "norm2"):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resBB is not None:
            sc = resBB
        elif resAB is not None:
            sc = resAB
        else:
            sc = None

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
    ) -> torch.Tensor:
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
            )
        conv_weights = self.edge_info(edge_feats)
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
    ) -> torch.Tensor:
        return self.rejector(
            node_feats,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_cutoff=edge_cutoff,
            radial_network=radial_network,
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
    ) -> torch.Tensor:
        if self.use_eqx and self.rejector.eqx_tp is not None and node_feats.is_cuda:
            conv_weights, radial_network = (
                radial_features(edge_feats),
                self.edge_info.mlp,
            )
        else:
            conv_weights, radial_network = self.edge_info(edge_feats), None
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
    ) -> torch.Tensor:
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
            )
        conv_weights = self.edge_info(edge_feats)
        return self.rejector(
            node_feats,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_cutoff,
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
