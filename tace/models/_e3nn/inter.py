################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum
from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import MLP, ScaledSigmoid, ScaledSiLU
from ..lammps import Graph
from .base import Interaction, _to_possible_tp_irreps
from .residual import get_resnet_layer
from .nonlinear import get_nonlinear_layer
from .layer_norm import get_normalization_layer
from .fused import O3ScatterTensorProduct, uuSO2ScatterTensorProduct, uvSO2TensorProduct
from .wigner6j import O3Wigner6jScatterTensorProduct
from .o2 import O2MagneticScatterLinear



class O3CgtpInteraction(Interaction):
    """
    An interaction module based on Clebsch-Gordan tensor products (CGTP).

    This module performs edge-level convolution using Clebsch-Gordan tensor
    products. It supports operator fusion via OpenEquivariance or CuEquivariance,
    which can significantly reduce memory consumption and improve efficiency.

    This interaction block does not directly add nonlinearity to the edge.
    """

    def _prepare_setup(self) -> None:
        pass

    def _build_rejector(self) -> torch.nn.Module:
        return O3ScatterTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return self.rejector.irreps_out.simplify()

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
            self.nonlinear_type,
            o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out]),
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

        self.edge_info = MLP(
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )

        if self._uses_edge_density():
            self.edge_density = MLP(
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act="silu",
            )  # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

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
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ["AB", "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
        magnetic_node_attrs: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        conv_weights = self.edge_info(edge_feats)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(node_feats, edge_attrs, conv_weights, edge_index)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        magnetic_radial_basis: Union[torch.Tensor, None],
        magnetic_node_attrs: Union[torch.Tensor, None],
        batch: torch.Tensor,
        graph: Graph,
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, "resnetBB"):
            if self.resnet_linear_type == "aware":
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats)

        if hasattr(self, "resnetBA"):
            if self.resnet_linear_type == "aware":
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, "norm1"):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        m_i = self.linear_down(
            self.truncate_ghosts(
                self._compute_messages(
                    node_feats,
                    node_attrs_total,
                    edge_radial_basis,
                    edge_feats,
                    edge_attrs,
                    edge_index,
                    edge_cutoff,
                    edge_wigner,
                    edge_wigner_inv,
                    magnetic_radial_basis,
                    magnetic_node_attrs,
                ),
                nlocal,
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if edge_cutoff is not None and self.apply_density_cutoff:
                density = density * edge_cutoff
            density = scatter_sum(
                density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0)
            )
            density = self.truncate_ghosts(density, nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        m_i = self._normalize_messages(m_i, density)

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        if resBA is not None:
            if hasattr(self, "stochastic_depth"):
                m_i = self.stochastic_depth(m_i, batch)
            m_i = m_i + resBA

        if hasattr(self, "resnetAB"):
            if self.resnet_linear_type == "aware":
                resAB = self.resnetAB(m_i, node_attrs_slice)
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


# It will be covered by future O2 version
class uuSO2Interaction(O3CgtpInteraction):
    """
    An interaction module based on uuSO2Linear.

    It achieves the same accuracy and extrapolation capability as CGTP.
    Set `export TACE_USE_EQX=1` to use the current EQX preview operator.

    This interaction block does not directly add nonlinearity to the message.
    """

    def _prepare_setup(self) -> None:
        assert self.parity == False, (
            "uuSO2InteractionArchitecture1 not support O(3) group"
        )
        assert self.irreps_in.lmax > 0, (
            "uuSO2InteractionArchitecture1's irreps_in.lmax must > 0, "
            "use uuSO2InteractionArchitecture1 from the second layer or use other node_embedding with l > 0"
        )
        assert self.edge_nonlinear == None

    def _build_rejector(self) -> torch.nn.Module:
        return uuSO2ScatterTensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            weight_type=self.so2_linear_type,
            l1l3=self.so2_l1l3,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(self.irreps_out),
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return self.irreps_in

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
        magnetic_node_attrs: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        conv_weights = self.edge_info(edge_feats)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(
            node_feats,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
        )


# A little BUG
class uvSO2Interaction(O3CgtpInteraction):
    """
    An interaction module based on uvSO2Linear,
    Edge Cluster Expansion and Radial Rotary Attention.

    It achieves highest accuracy at the expanse of computational efficiency.

    This interaction block add nonlinearity to the message.
    """

    def _prepare_setup(self) -> None:
        assert self.parity == False, "uvSO2Interaction not support O(3) group"
        assert self.irreps_in.lmax > 0, (
            "uvSO2Interaction's irreps_in.lmax must > 0, "
            "use uvSO2Interaction from the second layer or use other node_embedding with l > 0"
        )
        assert (
            self.edge_nonlinear == "so2_sigmoid_gate"
            or self.edge_nonlinear == "so2_silu_gate"
        )
        self.scatter_norm = None

    def _build_rejector(self) -> torch.nn.Module:
        edge_act = self.edge_nonlinear.split("_")[1]
        scalar_act = self.scalar_act or edge_act
        tensor_act = self.tensor_act or edge_act
        return uvSO2TensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            num_radial_basis=self.num_radial_basis,
            num_head=self.num_head,
            use_temperature=self.use_temperature,
            edge_ace_hidden=self.edge_ace_hidden,
            edge_wise_hidden=self.edge_wise_hidden,
            so2_linear_type=self.so2_linear_type,
            gate_m0=self.gate_m0,
            use_so2_edge_ace=self.use_so2_edge_ace,
            use_graph_softmax=self.use_graph_softmax,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(
                o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out])
            ),
            scalar_act=ScaledSigmoid() if scalar_act == "sigmoid" else ScaledSiLU(),
            tensor_act=ScaledSigmoid() if tensor_act == "sigmoid" else ScaledSiLU(),
            use_radial_phase=self.use_radial_phase,
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return o3.Irreps(
            [(self.edge_wise_hidden, ir) for _, ir in self.irreps_out]
        )

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
        magnetic_node_attrs: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        return self.rejector(
            node_feats,
            self.edge_info(edge_feats),
            edge_index,
            edge_cutoff,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
        )


# In dev TODO
class O3Wigner6jMagneticInteraction(O3CgtpInteraction):
    """O(3) magnetic interaction evaluated in a Wigner-6j recoupled tree.

    Positional and magnetic path weights are edge dependent, and both weight
    networks receive the source atom's magnetic radial basis in addition to the
    edge features.

    The tensor-product order is node @ magnetic first and spherical harmonic
    second, while remaining exactly equivalent to
    (node @ spherical harmonic) @ magnetic.
    """

    weight_level = "edge"
    use_magnetic_edge_feats = True
    use_message_magnetic_tensor_product = True

    def _setup(self) -> None:
        if self.magnetic_irreps is None:
            raise ValueError("o3_w6j_mag requires magnetic_irreps.")
        if not 1 <= self.mag_Lmax <= self.Lmax:
            raise ValueError("mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax.")
        if self.magnetic_irreps.lmax != self.mag_Lmax:
            raise ValueError("magnetic_irreps must end at mag_Lmax.")
        if any(multiplicity != 1 for multiplicity, _ in self.magnetic_irreps):
            raise ValueError("magnetic_irreps solid harmonics require multiplicity one")
        if any(irrep.p != 1 for _, irrep in self.magnetic_irreps):
            raise ValueError("A 1e magnetic vector only generates le solid harmonics")
        if not self.parity:
            raise ValueError(
                "wigner6j_magnetic_conv requires parity: true for full O(3)"
            )
        if self.weight_level not in {"edge", "node"}:
            raise ValueError(
                "weight_level must be either 'edge' or 'node', "
                f"got {self.weight_level!r}"
            )

        if self.use_message_magnetic_tensor_product:
            message_lmax = self.Lmax if self.correlation == 1 else self.lmax
            intermediate_irreps = _to_possible_tp_irreps(
                self.irreps_in,
                self.irreps_sh,
                self.parity,
                lmax=message_lmax + self.magnetic_irreps.lmax,
            )
            self.irrreps_tp_out = _to_possible_tp_irreps(
                intermediate_irreps,
                self.magnetic_irreps,
                self.parity,
                lmax=message_lmax,
            )
            self.irreps_out = (self.irrreps_tp_out * self.num_channel).regroup()

            if self.layer != self.num_layers - 1:
                intermediate_irreps = _to_possible_tp_irreps(
                    self.irreps_in,
                    self.irreps_sh,
                    self.parity,
                    lmax=self.Lmax + self.magnetic_irreps.lmax,
                )
                self.irreps_sc = _to_possible_tp_irreps(
                    intermediate_irreps,
                    self.magnetic_irreps,
                    self.parity,
                    lmax=self.Lmax,
                )
                self.irreps_sc = (self.irreps_sc * self.num_channel).regroup()

        super()._setup()

        magnetic_radial_dim = self.num_mag_radial_basis
        edge_weight_input_dim = self.edge_feats_channel
        if self.use_magnetic_edge_feats:
            edge_weight_input_dim += magnetic_radial_dim

        if not self.use_message_magnetic_tensor_product:
            self.edge_info = MLP(
                [edge_weight_input_dim]
                + self.radial_mlp
                + [self.rejector.weight_numel],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act="silu",
            )
            return

        self.rejector = self._build_magnetic_rejector()
        self.linear_down = e3nnLinear(
            self.rejector.irreps_out.simplify(),
            self.linear_down.irreps_out,
            bias=self.use_bias,
        )
        self.edge_info = MLP(
            [edge_weight_input_dim]
            + self.radial_mlp
            + [self.rejector.edge_weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )
        magnetic_weight_input_dim = (
            edge_weight_input_dim
            if self.weight_level == "edge"
            else magnetic_radial_dim
        )
        self.magnetic_info = MLP(
            [magnetic_weight_input_dim]
            + self.radial_mlp
            + [self.rejector.extra_weight_numel],
            bias=self.radial_bias,
            # layer_norm=self.radial_layer_norm, # TODO
            act="silu",
        )

    def _build_magnetic_rejector(self) -> torch.nn.Module:
        return O3Wigner6jScatterTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            extra_irreps_node_attrs=self.magnetic_irreps,
            l1l2=self.l1l2,
            weight_level=self.weight_level,
        )

    def _apply_magnetic_rejector(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        magnetic_node_attrs: torch.Tensor,
        radial_weights: torch.Tensor,
        magnetic_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return self.rejector(
            node_feats,
            edge_attrs,
            magnetic_node_attrs,
            radial_weights,
            magnetic_weights,
            edge_index,
        )

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
        magnetic_node_attrs: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        magnetic_input_required = (
            self.use_magnetic_edge_feats or self.use_message_magnetic_tensor_product
        )
        if magnetic_input_required and magnetic_radial_basis is None:
            raise ValueError(
                f"{type(self).__name__} requires precomputed magnetic_radial_basis"
            )
        if self.use_message_magnetic_tensor_product and magnetic_node_attrs is None:
            raise ValueError(
                f"{type(self).__name__} requires precomputed magnetic_node_attrs"
            )

        magnetic_edge_feats = edge_feats
        if self.use_magnetic_edge_feats:
            magnetic_edge_feats = torch.cat(
                [edge_feats, magnetic_radial_basis[edge_index[0]]],
                dim=-1,
            )

        radial_weights = self.edge_info(magnetic_edge_feats)
        if edge_cutoff is not None:
            radial_weights = radial_weights * edge_cutoff

        if not self.use_message_magnetic_tensor_product:
            return self.rejector(
                node_feats,
                edge_attrs,
                radial_weights,
                edge_index,
            )

        if self.weight_level == "edge":
            magnetic_weights = self.magnetic_info(magnetic_edge_feats)
        else:
            magnetic_node_feats = self.magnetic_info(magnetic_radial_basis)
            magnetic_weights = magnetic_node_feats
        return self._apply_magnetic_rejector(
            node_feats,
            edge_attrs,
            magnetic_node_attrs,
            radial_weights,
            magnetic_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
        )


# In dev TODO
class O2MagneticInteraction(O3CgtpInteraction):
    """Full-O3 magnetic interaction executed with local O2 linears."""

    path_mode = "uv"

    def _setup(self) -> None:
        if not self.mmax == self.Lmax == self.lmax:
            raise ValueError("o2_mag requires mmax == Lmax == lmax.")
        if not self.parity:
            raise ValueError("o2_mag requires parity: true for full O(3)")
        if self.path_mode not in {"uv", "uu"}:
            raise ValueError("O2 magnetic path_mode must be 'uv' or 'uu'.")
        if self.magnetic_irreps is None:
            raise ValueError("o2_mag requires magnetic_irreps.")
        if not 1 <= self.mag_Lmax <= self.Lmax:
            raise ValueError("mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax.")
        if self.magnetic_irreps.lmax != self.mag_Lmax:
            raise ValueError("magnetic_irreps must end at mag_Lmax.")

        super()._setup()
        self.rejector = O2MagneticScatterLinear(
            self.irreps_in,
            self.irreps_out,
            self.magnetic_irreps,
            num_channel=self.num_channel,
            lmax=self.lmax,
            path_mode=self.path_mode,
        )
        self.linear_down = e3nnLinear(
            self.rejector.irreps_out.simplify(),
            self.linear_down.irreps_out,
            bias=self.use_bias,
        )
        edge_weight_input_dim = self.edge_feats_channel + 2 * self.num_mag_radial_basis
        self.edge_info = MLP(
            [edge_weight_input_dim] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )

    def _compute_messages(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
        magnetic_node_attrs: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if magnetic_radial_basis is None:
            raise ValueError(
                "O2MagneticInteraction requires precomputed magnetic_radial_basis"
            )
        if magnetic_node_attrs is None:
            raise ValueError(
                "O2MagneticInteraction requires precomputed magnetic_node_attrs"
            )

        source, target = edge_index
        magnetic_edge_feats = torch.cat(
            (
                edge_feats,
                magnetic_radial_basis[source],
                magnetic_radial_basis[target],
            ),
            dim=-1,
        )
        radial_weights = self.edge_info(magnetic_edge_feats)
        if edge_cutoff is not None:
            radial_weights = radial_weights * edge_cutoff
        return self.rejector(
            node_feats,
            magnetic_node_attrs,
            radial_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
        )


INTERACTION: Dict[str, type[Interaction]] = {
    "normal": O3CgtpInteraction,
    "spectral": O3CgtpInteraction,
    "cgtp": O3CgtpInteraction,
    "uu_so2": uuSO2Interaction,
    "so2": uvSO2Interaction,
    "uv_so2": uvSO2Interaction,
    "attn": uvSO2Interaction,
    "o3_w6j_mag": O3Wigner6jMagneticInteraction,
    "o2_mag": O2MagneticInteraction,
}
