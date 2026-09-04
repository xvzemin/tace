################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from ..lammps import Graph
from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import ACTIVATION, MLP, get_scaled_activation
from .base import Interaction
from .fused import O3ScatterTensorProduct
from .layer_norm import get_normalization_layer
from .legacy_so2 import uvSO2Convolution
from .nonlinear import get_nonlinear_layer
from .o2 import O2ScatterMagneticTensorProduct, O2ScatterTensorProduct
from .residual import get_resnet_layer


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

    def _edge_weight_input_dim(self) -> int:
        return self.edge_feats_channel

    def _edge_weight_numel(self) -> int:
        return self.rejector.weight_numel

    def _setup_additional_modules(self) -> None:
        pass

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

        self.edge_info = MLP(
            [self._edge_weight_input_dim()]
            + self.radial_mlp
            + [self._edge_weight_numel()],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )
        self._setup_additional_modules()

        if self._uses_edge_density():
            self.edge_density = MLP(
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act="silu",
            )
            self.alpha = torch.nn.Parameter(
                torch.tensor(self.avg_num_neighbors**0.5)
            )
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
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_edge_feats: Union[torch.Tensor, None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
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
        magnetic_edge_feats: Union[torch.Tensor, None],
        magnetic_edge_attrs: Union[torch.Tensor, None],
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
                    magnetic_edge_feats,
                    magnetic_edge_attrs,
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


class uvSO2Interaction(O3CgtpInteraction):
    """
    An interaction module based on uvSO2Linear,
    Edge Cluster Expansion and Radial Rotary Attention.

    It achieves highest accuracy at the expanse of computational efficiency.

    This interaction block add nonlinearity to the message.
    """

    use_temperature = True
    so2_linear_type = "w1"  # w1: O(3), w1_w1, w1_w2: SO(3)
    use_radial_phase = True # False: O(3), True:  SO(3)

    def _prepare_setup(self) -> None:
        super()._prepare_setup()
        if self.parity:
            raise ValueError("uvSO2Interaction does not support O(3).")
        if self.irreps_in.lmax <= 0:
            raise ValueError(
                "uvSO2Interaction requires irreps_in.lmax > 0. Use it after "
                "the first layer or provide a node embedding with l > 0."
            )
        if self.edge_nonlinear is None:
            raise ValueError("uvSO2Interaction requires edge_nonlinear to be set.")
        self.scatter_norm = None

    def _build_rejector(self) -> torch.nn.Module:
        if self.scalar_act is None:
            scalar_act = "silu"
        elif isinstance(self.scalar_act, str):
            scalar_act = self.scalar_act
        elif isinstance(self.scalar_act, list) and len(self.scalar_act) == 2:
            if not all(isinstance(name, str) for name in self.scalar_act):
                raise TypeError("scalar_act entries must be strings.")
            scalar_act = self.scalar_act[0]
        else:
            raise TypeError(
                "scalar_act must be None, a string, or a list of two strings "
                "for 0e and 0o."
            )
        tensor_act = self.tensor_act or (
            "silu" if self.use_asymmetric_contraction else "sigmoid"
        )
        if not isinstance(tensor_act, str):
            raise TypeError("tensor_act must be None or a string for tensor gates.")
        return uvSO2Convolution(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            num_radial_basis=self.num_radial_basis,
            num_head=self.num_head,
            use_temperature=self.use_temperature,
            edge_ace_hidden=self.edge_ace_hidden,
            so2_linear_type=self.so2_linear_type,
            gate_m0=False,
            use_asymmetric_contraction=self.use_asymmetric_contraction,
            use_radial_rotary_attention=self.use_radial_rotary_attention,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(
                o3.Irreps([(self.num_channel, ir) for _, ir in self.irreps_out])
            ),
            scalar_act=get_scaled_activation(scalar_act),
            tensor_act=get_scaled_activation(tensor_act),
            use_radial_phase=self.use_radial_phase,
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return o3.Irreps([(self.num_channel, ir) for _, ir in self.irreps_out])

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
        magnetic_edge_feats: Union[torch.Tensor, None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
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


class O2Interaction(O3CgtpInteraction):
    """Global O3 interaction evaluated through local O2 irreps.

    The interaction concatenates source and target node features and applies
    ``linear -> gate -> linear`` before inverse rotation and scatter. Optional
    radial rotary attention uses radial basis as scale and shift.
    """

    def _build_rejector(self) -> torch.nn.Module:
        rejector = O2ScatterTensorProduct(
            self.irreps_in,
            self.irreps_out,
            num_channel=self.num_channel,
            lmax=max(self.Lmax, self.lmax),
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
                    self.scalar_act[1]
                    if isinstance(self.scalar_act, list)
                    else "tanh"
                ).removeprefix("scaled_")
            ](),
            tensor_act=ACTIVATION[
                (self.tensor_act or "sigmoid").removeprefix("scaled_")
            ](),
            num_head=self.num_head,
            num_radial_basis=self.num_radial_basis,
            use_radial_rotary_attention=self.use_radial_rotary_attention,
        )
        if rejector.attention is not None:
            self.scatter_norm = None
        return rejector

    def _edge_weight_input_dim(self) -> int:
        return self.edge_feats_channel

    def _edge_weight_inputs(
        self,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        magnetic_edge_feats: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return edge_feats

    def _apply_rejector(
        self,
        node_feats: torch.Tensor,
        magnetic_edge_feats: Union[torch.Tensor, None],
        magnetic_edge_attrs: Union[torch.Tensor, None],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return self.rejector(
            node_feats,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis=edge_radial_basis,
            edge_cutoff=edge_cutoff,
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
        node_attrs_total: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
        edge_wigner: Union[torch.Tensor, None] = None,
        edge_wigner_inv: Union[torch.Tensor, None] = None,
        magnetic_edge_feats: Union[torch.Tensor, None] = None,
        magnetic_edge_attrs: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        edge_weight_inputs = self._edge_weight_inputs(
            edge_feats,
            edge_index,
            magnetic_edge_feats,
        )
        conv_weights = self.edge_info(edge_weight_inputs)
        return self._apply_rejector(
            node_feats,
            magnetic_edge_feats,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
            edge_cutoff,
        )


class O2MagneticInteraction(O2Interaction):
    """Local-O2 interaction augmented by magnetic edge attributes."""

    def _prepare_setup(self) -> None:
        if not self.parity:
            raise ValueError("o2_mag requires parity: true for full O(3)")
        if self.magnetic_edge_irreps is None:
            raise ValueError("o2_mag requires magnetic_edge_irreps.")
        if not 1 <= self.mag_Lmax <= self.Lmax:
            raise ValueError("mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax.")
        if self.magnetic_edge_irreps.lmax != self.mag_Lmax:
            raise ValueError("magnetic_edge_irreps must end at mag_Lmax.")
        self.magnetic_edge_irreps_out = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.magnetic_edge_irreps]
        ).regroup()
        super()._prepare_setup()

    def _build_rejector(self) -> torch.nn.Module:
        rejector = O2ScatterMagneticTensorProduct(
            self.irreps_in,
            self.irreps_out,
            self.magnetic_edge_irreps_out,
            num_channel=self.num_channel,
            lmax=max(self.Lmax, self.lmax),
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
                    self.scalar_act[1]
                    if isinstance(self.scalar_act, list)
                    else "tanh"
                ).removeprefix("scaled_")
            ](),
            tensor_act=ACTIVATION[
                (self.tensor_act or "sigmoid").removeprefix("scaled_")
            ](),
            num_head=self.num_head,
            num_radial_basis=self.num_radial_basis,
            use_radial_rotary_attention=self.use_radial_rotary_attention,
        )
        if rejector.attention is not None:
            self.scatter_norm = None
        return rejector

    def _setup_additional_modules(self) -> None:
        self.magnetic_linear = e3nnLinear(
            self.magnetic_edge_irreps,
            self.magnetic_edge_irreps_out,
            bias=False,
            internal_weights=False,
        )
        self.magnetic_edge_info = MLP(
            [self.magnetic_edge_feats_channel]
            + self.radial_mlp
            + [self.magnetic_linear.weight_numel],
            bias=self.radial_bias,
            # layer_norm=self.radial_layer_norm,
            act="silu",
        )
    def _apply_rejector(
        self,
        node_feats: torch.Tensor,
        magnetic_edge_feats: Union[torch.Tensor, None],
        magnetic_edge_attrs: Union[torch.Tensor, None],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if magnetic_edge_feats is None or magnetic_edge_attrs is None:
            raise ValueError(
                "O2MagneticInteraction requires magnetic edge features and attrs"
            )
        magnetic_edge_attrs = self.magnetic_linear(
            magnetic_edge_attrs,
            self.magnetic_edge_info(magnetic_edge_feats),
        )
        return self.rejector(
            node_feats,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
            edge_cutoff,
        )


INTERACTION: Dict[str, type[Interaction]] = {
    "cgtp": O3CgtpInteraction,
    "so2": uvSO2Interaction,
    # below in dev, not stable
    "o2": O2Interaction,
    "o2_mag": O2MagneticInteraction,
}

interaction_aliases = {
    "normal": "cgtp",
    "spectral": "cgtp",
    "uv_so2": "so2",
    "attn": "so2",
}

INTERACTION.update(
    {
        alias: INTERACTION[target]
        for alias, target in interaction_aliases.items()
    }
)
