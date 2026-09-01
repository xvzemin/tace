################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch

from eqx import co3
from tace.utils.torch_scatter import scatter_sum

from ..mlp import MLP
from .base import Interaction, possible_irreps
from .dropout import GraphDropPath
from .fused import O3MagneticScatterTensorProduct, O3ScatterTensorProduct
from .linear import Linear
from .nonlinear import CartesianGate, split_scalars
from .residual import get_resnet_layer


class CartesianCgtpInteraction(Interaction):
    """Cartesian counterpart of the standard CG tensor-product interaction."""

    def _setup(self) -> None:
        self.irreps_up = co3.Irreps(
            [(ir, self.num_channel) for ir, _ in self.irreps_in]
        ).regroup()
        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_up,
            bias=self.use_bias,
        )
        self.rejector = self._build_rejector()
        self.edge_info = MLP(
            [
                self._edge_weight_input_dim(),
                *self.radial_mlp,
                self.rejector.weight_numel,
            ],
            bias=self.radial_bias,
            layer_norm=self.edge_feats_channel != self.num_radial_basis,
            act="silu",
        )
        self._setup_additional_modules()
        self._setup_output()

    def _build_rejector(self) -> torch.nn.Module:
        return O3ScatterTensorProduct(
            self.irreps_up,
            self.irreps_edge,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _edge_weight_input_dim(self) -> int:
        return self.edge_feats_channel

    def _setup_additional_modules(self) -> None:
        pass

    def _compute_messages(
        self,
        lifted: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Optional[torch.Tensor],
        magnetic_radial_basis,
        magnetic_node_attrs,
    ) -> torch.Tensor:
        conv_weights = self.edge_info(edge_feats)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(lifted, edge_attrs, conv_weights, edge_index)

    def _setup_output(self) -> None:
        if self.nonlinear == "gate":
            scalars, tensors = split_scalars(self.irreps_out)
            self.nonlinearity = CartesianGate(
                scalars,
                tensors,
                scalar_act=self.scalar_act,
                tensor_act=self.tensor_act,
            )
            linear_out = self.nonlinearity.irreps_in
        elif self.nonlinear is None:
            self.nonlinearity = None
            linear_out = self.irreps_out
        else:
            raise ValueError(
                "Cartesian interactions support nonlinear=None or 'gate', "
                f"got {self.nonlinear!r}."
            )
        self.linear_down = Linear(self.irreps_out, linear_out, bias=self.use_bias)
        use_residual = self.layer > 0 or self.use_first_resnet
        if use_residual and self.resnet_type == "BB":
            self.residual_bb = get_resnet_layer(
                self.irreps_in,
                self.irreps_sc,
                self.use_bias,
                self.num_elements,
                self.resnet_linear_type,
            )
        if use_residual and self.resnet_type == "BAB":
            self.residual_ba = get_resnet_layer(
                self.irreps_in,
                self.irreps_out,
                self.use_bias,
                self.num_elements,
                self.resnet_linear_type,
            )
            if self.stochastic_depth_p:
                self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)
        if use_residual and self.resnet_type in {"AB", "BAB"}:
            self.residual_ab = get_resnet_layer(
                self.irreps_out,
                self.irreps_sc,
                self.use_bias,
                self.num_elements,
                self.resnet_linear_type,
            )
        if (
            self.layer > 0 or self.use_first_pre_norm
        ) and self.pre_norm_type is not None:
            from .layer_norm import get_normalization_layer

            if self.resnet_type in {"BB", "BAB"}:
                self.norm_in = get_normalization_layer(
                    self.pre_norm_type, self.irreps_in
                )
            if self.resnet_type in {"AB", "BAB"}:
                self.norm_out = get_normalization_layer(
                    self.pre_norm_type, self.irreps_out
                )
        if self._uses_edge_density():
            self.edge_density = MLP(
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.edge_feats_channel != self.num_radial_basis,
                act="silu",
            )
            self.density_scale = torch.nn.Parameter(
                torch.tensor(self.avg_num_neighbors**0.5)
            )
            self.density_shift = torch.nn.Parameter(torch.tensor(0.0))

    def _density(
        self,
        edge_feats: torch.Tensor,
        edge_cutoff: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self._uses_edge_density():
            return None
        values = torch.tanh(self.edge_density(edge_feats).square())
        if edge_cutoff is not None and self.apply_density_cutoff:
            values = values * edge_cutoff
        density = scatter_sum(values, edge_index[1], dim=0, dim_size=num_nodes)
        density = density * self.density_shift + self.density_scale
        return density.masked_fill(density == 0, 1.0e-9)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Optional[torch.Tensor],
        edge_wigner=None,
        edge_wigner_inv=None,
        magnetic_radial_basis=None,
        magnetic_node_attrs=None,
        batch=None,
        graph=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if graph is not None:
            node_feats = self.handle_lammps(
                node_feats,
                graph.lmp_data,
                graph.lmp_natoms,
                self.layer,
            )
        residual_input = node_feats
        residual_bb = None
        residual_ba = None
        if hasattr(self, "residual_bb"):
            residual_bb = (
                self.residual_bb(residual_input, node_attrs)
                if self.resnet_linear_type == "aware"
                else self.residual_bb(residual_input)
            )
        if hasattr(self, "residual_ba"):
            residual_ba = (
                self.residual_ba(residual_input, node_attrs)
                if self.resnet_linear_type == "aware"
                else self.residual_ba(residual_input)
            )
        if hasattr(self, "norm_in"):
            node_feats = self.norm_in(node_feats)
        lifted = self.linear_up(node_feats)
        messages = self._compute_messages(
            lifted,
            edge_feats,
            edge_attrs,
            edge_index,
            edge_cutoff,
            magnetic_radial_basis,
            magnetic_node_attrs,
        )
        density = self._density(
            edge_feats,
            edge_cutoff,
            edge_index,
            messages.size(0),
            messages.dtype,
            messages.device,
        )
        messages = self._normalize_messages(messages, density)
        messages = self.linear_down(messages)
        if self.nonlinearity is not None:
            messages = self.nonlinearity(messages)
        if residual_ba is not None:
            if hasattr(self, "stochastic_depth"):
                residual_ba = self.stochastic_depth(residual_ba, batch)
            messages = messages + residual_ba
        if hasattr(self, "norm_out"):
            messages = self.norm_out(messages)
        nlocal = graph.lmp_natoms[0] if graph is not None and graph.lmp else None
        messages = self.truncate_ghosts(messages, nlocal)
        if residual_bb is not None:
            sc = residual_bb
        elif hasattr(self, "residual_ab"):
            sc = (
                self.residual_ab(messages, node_attrs)
                if self.resnet_linear_type == "aware"
                else self.residual_ab(messages)
            )
        else:
            sc = None
        sc = self.truncate_ghosts(sc, nlocal)
        return messages, sc


class CartesianMagneticInteraction(CartesianCgtpInteraction):
    """Three-factor Cartesian interaction with axial magnetic harmonics."""

    def _setup(self) -> None:
        if self.magnetic_irreps is None:
            raise ValueError("o3_w6j_mag requires magnetic Cartesian irreps.")
        message_lmax = self.Lmax if self.correlation == 1 else self.lmax
        intermediate = possible_irreps(
            self.irreps_in,
            self.irreps_edge,
            parity=True,
            lmax=message_lmax + self.magnetic_irreps.lmax,
        )
        output_types = possible_irreps(
            intermediate,
            self.magnetic_irreps,
            parity=True,
            lmax=message_lmax,
        )
        self.irreps_out = co3.Irreps([(ir, self.num_channel) for ir, _ in output_types])
        if self.layer != self.num_layers - 1:
            self.irreps_sc = self.irreps_out.filter(lmax=self.Lmax)
        super()._setup()

    def _build_rejector(self) -> torch.nn.Module:
        if not self.parity:
            raise ValueError("o3_w6j_mag requires parity=true.")
        return O3MagneticScatterTensorProduct(
            self.irreps_up,
            self.irreps_edge,
            self.magnetic_irreps,
            self.irreps_out,
            l1l2=self.l1l2,
        )

    def _edge_weight_input_dim(self) -> int:
        return self.edge_feats_channel + self.num_mag_radial_basis

    def _setup_additional_modules(self) -> None:
        self.extra_info = MLP(
            [
                self._edge_weight_input_dim(),
                *self.radial_mlp,
                self.rejector.extra_weight_numel,
            ],
            bias=self.radial_bias,
            layer_norm=True,
            act="silu",
        )

    def _compute_messages(
        self,
        lifted: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: Optional[torch.Tensor],
        magnetic_radial_basis,
        magnetic_node_attrs,
    ) -> torch.Tensor:
        if magnetic_radial_basis is None or magnetic_node_attrs is None:
            raise ValueError(
                "o3_w6j_mag requires magnetic radial and angular features."
            )
        source = edge_index[0]
        inputs = torch.cat((edge_feats, magnetic_radial_basis[source]), dim=-1)
        conv_weights = self.edge_info(inputs)
        extra_weights = self.extra_info(inputs)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
            extra_weights = extra_weights * edge_cutoff
        return self.rejector(
            lifted,
            edge_attrs,
            magnetic_node_attrs,
            conv_weights,
            extra_weights,
            edge_index,
        )


INTERACTION = {
    "cgtp": CartesianCgtpInteraction,
    "spatial": CartesianCgtpInteraction,
    "coupled": CartesianCgtpInteraction,
    "o3_w6j_mag": CartesianMagneticInteraction,
}


__all__ = [
    "CartesianCgtpInteraction",
    "CartesianMagneticInteraction",
    "INTERACTION",
]
