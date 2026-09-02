################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch

from eqx import co3
from tace.utils.torch_scatter import scatter_sum

from ..mlp import MLP, get_scaled_activation
from .base import Interaction
from .dropout import GraphDropPath
from .fused import O3ScatterTensorProduct
from .linear import Linear
from .nonlinear import Gate, O3Gate, split_scalars
from .residual import get_resnet_layer


class O3IctpInteraction(Interaction):
    """Interaction based on irreducible tensor products."""

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
    ) -> torch.Tensor:
        conv_weights = self.edge_info(edge_feats)
        if edge_cutoff is not None:
            conv_weights = conv_weights * edge_cutoff
        return self.rejector(lifted, edge_attrs, conv_weights, edge_index)

    def _setup_output(self) -> None:
        if self.nonlinear == "gate":
            if self.gate_m0:
                scalar_name = (
                    self.scalar_act[0]
                    if isinstance(self.scalar_act, list)
                    else self.scalar_act
                )
                irreps_gates = co3.Irreps(
                    [(co3.Irrep("0e"), mul) for _, mul in self.irreps_out]
                )
                self.nonlinearity = O3Gate(
                    irreps_gates,
                    [get_scaled_activation(scalar_name or "sigmoid")]
                    * len(irreps_gates),
                    self.irreps_out,
                )
            else:
                scalars, tensors = split_scalars(self.irreps_out)
                self.nonlinearity = Gate(
                    scalars,
                    tensors,
                    scalar_act=self.scalar_act,
                    tensor_act=self.tensor_act,
                )
            linear_out = self.nonlinearity.irreps_in
            self.linear_nonlinearity = Linear(
                self.irreps_out if self.gate_m0 else self.nonlinearity.irreps_out,
                self.irreps_out,
                bias=self.use_bias,
            )
        elif self.nonlinear is None:
            self.nonlinearity = None
            linear_out = self.irreps_out
            self.linear_nonlinearity = torch.nn.Identity()
        else:
            raise ValueError(
                "Interactions support nonlinear=None or 'gate', "
                f"got {self.nonlinear!r}."
            )
        self.linear_down = Linear(
            self.rejector.irreps_out,
            linear_out,
            bias=self.use_bias,
        )
        self.irreps_project = linear_out
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
        )
        messages = self.linear_down(messages)
        messages = co3.project_irreps(messages, self.irreps_project)
        density = self._density(
            edge_feats,
            edge_cutoff,
            edge_index,
            messages.size(0),
            messages.dtype,
            messages.device,
        )
        messages = self._normalize_messages(messages, density)
        if self.nonlinearity is not None:
            messages = self.nonlinearity(messages)
        messages = self.linear_nonlinearity(messages)
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


INTERACTION = {
    "ictp": O3IctpInteraction,
}


__all__ = [
    "O3IctpInteraction",
    "INTERACTION",
]
