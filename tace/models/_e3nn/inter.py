################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import abc
from typing import Dict, Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum
from ..lammps import Graph
from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import MLP, get_scaled_activation
from .base import Interaction, _to_possible_tp_irreps
from .fused import O3ScatterTensorProduct
from .legacy_so2 import uvSO2Convolution
from .layer_norm import get_normalization_layer
from .nonlinear import get_nonlinear_layer
from .o2 import O2MagneticScatterLinear, O2ScatterLinear
from .residual import get_resnet_layer
from .wigner6j import O3Wigner6jScatterTensorProduct


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


class O3GeneralizedWigner6jInteraction(O3CgtpInteraction, abc.ABC):
    """Abstract three-factor O(3) interaction using Wigner-6j recoupling.

    The reference tensor-product order is ``(node @ edge) @ extra`` and the
    executed order is ``(node @ extra) @ edge``. Subclasses must define the
    extra node representation through :attr:`extra_irreps_node_attrs`.
    """

    weight_level = "edge"

    @property
    @abc.abstractmethod
    def extra_irreps_node_attrs(self) -> o3.Irreps:
        """Irreps of the additional equivariant node attributes."""
        raise NotImplementedError

    def _prepare_setup(self) -> None:
        super()._prepare_setup()
        extra_irreps_node_attrs = o3.Irreps(self.extra_irreps_node_attrs)
        if any(multiplicity != 1 for multiplicity, _ in extra_irreps_node_attrs):
            raise ValueError("extra_irreps_node_attrs must have multiplicity one")
        if self.weight_level not in {"edge", "node"}:
            raise ValueError(
                "weight_level must be either 'edge' or 'node', "
                f"got {self.weight_level!r}"
            )

        message_lmax = self.Lmax if self.correlation == 1 else self.lmax
        intermediate_irreps = _to_possible_tp_irreps(
            self.irreps_in,
            self.irreps_sh,
            self.parity,
            lmax=message_lmax + extra_irreps_node_attrs.lmax,
        )
        self.irrreps_tp_out = _to_possible_tp_irreps(
            intermediate_irreps,
            extra_irreps_node_attrs,
            self.parity,
            lmax=message_lmax,
        )
        self.irreps_out = (self.irrreps_tp_out * self.num_channel).regroup()

        if self.layer != self.num_layers - 1:
            intermediate_irreps = _to_possible_tp_irreps(
                self.irreps_in,
                self.irreps_sh,
                self.parity,
                lmax=self.Lmax + extra_irreps_node_attrs.lmax,
            )
            self.irreps_sc = _to_possible_tp_irreps(
                intermediate_irreps,
                extra_irreps_node_attrs,
                self.parity,
                lmax=self.Lmax,
            )
            self.irreps_sc = (self.irreps_sc * self.num_channel).regroup()

    def _build_rejector(self) -> torch.nn.Module:
        return O3Wigner6jScatterTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            extra_irreps_node_attrs=self.extra_irreps_node_attrs,
            weight_level=self.weight_level,
            l1l2=self.l1l2,
            register_reference=False,
        )

    def _extra_radial_basis_dim(self) -> int:
        return self.num_mag_radial_basis

    def _edge_weight_input_dim(self) -> int:
        return self.edge_feats_channel + self._extra_radial_basis_dim()

    def _edge_weight_numel(self) -> int:
        return self.rejector.edge_weight_numel

    def _setup_additional_modules(self) -> None:
        extra_weight_input_dim = (
            self._edge_weight_input_dim()
            if self.weight_level == "edge"
            else self._extra_radial_basis_dim()
        )
        self.extra_info = MLP(
            [extra_weight_input_dim]
            + self.radial_mlp
            + [self.rejector.extra_weight_numel],
            bias=self.radial_bias,
            act="silu",
        )

    def _resolve_extra_inputs(
        self,
        extra_radial_basis: Union[torch.Tensor, None],
        extra_node_attrs: Union[torch.Tensor, None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_radial_basis is None:
            raise ValueError(
                f"{type(self).__name__} requires precomputed extra_radial_basis"
            )
        if extra_node_attrs is None:
            raise ValueError(
                f"{type(self).__name__} requires precomputed extra_node_attrs"
            )
        return extra_radial_basis, extra_node_attrs

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
        extra_radial_basis, extra_node_attrs = self._resolve_extra_inputs(
            magnetic_radial_basis,
            magnetic_node_attrs,
        )

        extra_edge_feats = torch.cat(
            [edge_feats, extra_radial_basis[edge_index[0]]],
            dim=-1,
        )

        radial_weights = self.edge_info(extra_edge_feats)
        if edge_cutoff is not None:
            radial_weights = radial_weights * edge_cutoff

        if self.weight_level == "edge":
            extra_weights = self.extra_info(extra_edge_feats)
        else:
            extra_weights = self.extra_info(extra_radial_basis)

        return self.rejector(
            node_feats,
            edge_attrs,
            extra_node_attrs,
            radial_weights,
            extra_weights,
            edge_index,
        )


class uvSO2Interaction(O3CgtpInteraction):
    """
    An interaction module based on uvSO2Linear,
    Edge Cluster Expansion and Radial Rotary Attention.

    It achieves highest accuracy at the expanse of computational efficiency.

    This interaction block add nonlinearity to the message.
    """

    so2_linear_type = "w1"
    use_temperature = True
    use_so2_edge_ace = True
    use_graph_softmax = True
    use_radial_phase = True

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
        tensor_act = self.tensor_act or "sigmoid"
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
            edge_wise_hidden=self.edge_wise_hidden,
            so2_linear_type=self.so2_linear_type,
            gate_m0=False,
            use_so2_edge_ace=self.use_so2_edge_ace,
            use_graph_softmax=self.use_graph_softmax,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(
                o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out])
            ),
            scalar_act=get_scaled_activation(scalar_act),
            tensor_act=get_scaled_activation(tensor_act),
            use_radial_phase=self.use_radial_phase,
        )

    def _linear_down_irreps_in(self) -> o3.Irreps:
        return o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out])

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


class O2Interaction(O3CgtpInteraction):
    """Global O3 interaction evaluated through local O2 irreps.

    The default path concatenates source and target node features and applies
    ``uv -> gate -> uv`` before inverse rotation and scatter. The optional O2
    asymmetric contraction replaces the gate, with its scalar coefficients
    generated by the first O2 linear. Optional radial rotary attention uses
    radial basis as scale and shift.
    """

    def _build_rejector(self) -> torch.nn.Module:
        rejector = O2ScatterLinear(
            self.irreps_in,
            self.irreps_out,
            num_channel=self.num_channel,
            lmax=max(self.Lmax, self.lmax),
            mmax=self.mmax,
            act_0e=get_scaled_activation(self._o2_act_0e_name),
            act_0o=get_scaled_activation(self._o2_act_0o_name),
            act_lm=get_scaled_activation(self._o2_act_lm_name),
            correlation=2, # hardcore 2 now for memory
            num_head=self.num_head,
            num_radial_basis=self.num_radial_basis,
            use_asymmetric_contraction=self.use_asymmetric_contraction,
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
        magnetic_radial_basis: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return edge_feats

    def _apply_rejector(
        self,
        node_feats: torch.Tensor,
        magnetic_node_attrs: Union[torch.Tensor, None],
        radial_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return self.rejector(
            node_feats,
            radial_weights,
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
        self.use_radial_rotary_attention = (
            self.use_radial_rotary_attention
            and min(self.irreps_in.lmax, self.mmax) > 0
        )
        contraction_input_lmax = self.irreps_in.lmax
        if self.magnetic_irreps is not None:
            contraction_input_lmax = max(
                contraction_input_lmax,
                self.magnetic_irreps.lmax,
            )
        contraction_mmax = min(
            contraction_input_lmax,
            self.irreps_out.lmax,
            self.mmax,
        )
        self.use_asymmetric_contraction = (
            self.use_asymmetric_contraction
            and contraction_mmax > 0
        )

        scalar_act = self.scalar_act
        if scalar_act is None:
            act_0e_name = "silu"
            act_0o_name = "tanh"
        elif isinstance(scalar_act, str):
            act_0e_name = scalar_act
            act_0o_name = "tanh"
        elif isinstance(scalar_act, list) and len(scalar_act) == 2:
            act_0e_name, act_0o_name = scalar_act
            if not isinstance(act_0e_name, str) or not isinstance(act_0o_name, str):
                raise TypeError("O2 scalar_act entries must be strings.")
        else:
            raise TypeError(
                "O2 scalar_act must be None, a string, or a list of two strings "
                "for 0e and 0o."
            )
        act_lm_name = self.tensor_act or "sigmoid"
        if not isinstance(act_lm_name, str):
            raise TypeError("O2 tensor_act must be None or a string for lm gates.")

        self._o2_act_0e_name = act_0e_name
        self._o2_act_0o_name = act_0o_name
        self._o2_act_lm_name = act_lm_name

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
        edge_weight_inputs = self._edge_weight_inputs(
            edge_feats,
            edge_index,
            magnetic_radial_basis,
        )
        radial_weights = self.edge_info(edge_weight_inputs)
        return self._apply_rejector(
            node_feats,
            magnetic_node_attrs,
            radial_weights,
            edge_index,
            edge_wigner,
            edge_wigner_inv,
            edge_radial_basis,
            edge_cutoff,
        )


class O3Wigner6jMagneticInteraction(O3GeneralizedWigner6jInteraction):
    """Generalized Wigner-6j interaction for magnetic solid harmonics."""

    @property
    def extra_irreps_node_attrs(self) -> o3.Irreps:
        return self.magnetic_irreps

    def _prepare_setup(self) -> None:
        if self.magnetic_irreps is None:
            raise ValueError("o3_w6j_mag requires magnetic_irreps.")
        if not 1 <= self.mag_Lmax <= self.Lmax:
            raise ValueError("mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax.")
        if self.magnetic_irreps.lmax != self.mag_Lmax:
            raise ValueError("magnetic_irreps must end at mag_Lmax.")
        if any(irrep.p != 1 for _, irrep in self.magnetic_irreps):
            raise ValueError("A 1e magnetic vector only generates le solid harmonics")
        if not self.parity:
            raise ValueError(
                "wigner6j_magnetic_conv requires parity: true for full O(3)"
            )
        super()._prepare_setup()


class O2MagneticInteraction(O2Interaction):
    """Local-O2 interaction augmented by magnetic solid harmonics."""

    def _prepare_setup(self) -> None:
        if not self.parity:
            raise ValueError("o2_mag requires parity: true for full O(3)")
        if self.magnetic_irreps is None:
            raise ValueError("o2_mag requires magnetic_irreps.")
        if not 1 <= self.mag_Lmax <= self.Lmax:
            raise ValueError("mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax.")
        if self.magnetic_irreps.lmax != self.mag_Lmax:
            raise ValueError("magnetic_irreps must end at mag_Lmax.")
        super()._prepare_setup()

    def _build_rejector(self) -> torch.nn.Module:
        rejector = O2MagneticScatterLinear(
            self.irreps_in,
            self.irreps_out,
            self.magnetic_irreps,
            num_channel=self.num_channel,
            lmax=max(self.Lmax, self.lmax),
            mmax=self.mmax,
            act_0e=get_scaled_activation(self._o2_act_0e_name),
            act_0o=get_scaled_activation(self._o2_act_0o_name),
            act_lm=get_scaled_activation(self._o2_act_lm_name),
            correlation=self.correlation,
            num_head=self.num_head,
            num_radial_basis=self.num_radial_basis,
            use_asymmetric_contraction=self.use_asymmetric_contraction,
            use_radial_rotary_attention=self.use_radial_rotary_attention,
        )
        if rejector.attention is not None:
            self.scatter_norm = None
        return rejector

    def _edge_weight_input_dim(self) -> int:
        return self.edge_feats_channel + 2 * self.num_mag_radial_basis

    def _edge_weight_inputs(
        self,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        magnetic_radial_basis: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if magnetic_radial_basis is None:
            raise ValueError(
                "O2MagneticInteraction requires precomputed magnetic_radial_basis"
            )
        source, target = edge_index
        return torch.cat(
            (
                edge_feats,
                magnetic_radial_basis[source],
                magnetic_radial_basis[target],
            ),
            dim=-1,
        )

    def _apply_rejector(
        self,
        node_feats: torch.Tensor,
        magnetic_node_attrs: Union[torch.Tensor, None],
        radial_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_wigner: Union[torch.Tensor, None],
        edge_wigner_inv: Union[torch.Tensor, None],
        edge_radial_basis: torch.Tensor,
        edge_cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if magnetic_node_attrs is None:
            raise ValueError(
                "O2MagneticInteraction requires precomputed magnetic_node_attrs"
            )
        return self.rejector(
            node_feats,
            magnetic_node_attrs,
            radial_weights,
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
    "o3_w6j_mag": O3Wigner6jMagneticInteraction,
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
