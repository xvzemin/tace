################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

from ..linear import e3nnLinear
from ..mlp import MLP
from .base import EdgeInteraction
from .fused import UvO2TensorProduct, UvuTensorProduct
from .paths import generate_paths


class O3CgtpEdgeInteraction(EdgeInteraction):
    """Couple edge features with supplied spherical harmonics using CGTP."""

    def _setup(self) -> None:
        self.irreps_hidden = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_in.regroup()]
        )
        self._setup_convolution()

        radial_basis = self.radial_basis
        radial_norm = radial_basis["mlp_norm"] or "layer_norm"
        self.edge_info = MLP[self.mlp_type](
            [radial_basis["num_radial_basis"] + 2 * self.num_elements]
            + list(radial_basis["hidden"])
            + [self.weight_numel],
            bias=radial_basis["bias"],
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
        )

    def _setup_convolution(self) -> None:
        self.linear_up = e3nnLinear(
            self.irreps_in + self.irreps_in, self.irreps_hidden, bias=False
        )
        instructions, self.irreps_out = generate_paths(
            self.irreps_out, self.irreps_hidden, self.irreps_sh
        )
        self.rejector = UvuTensorProduct(
            self.irreps_hidden,
            self.irreps_sh,
            self.irreps_out,
            instructions,
            shared_weights=False,
        )
        self.weight_numel = self.rejector.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        edge_cutoff: torch.Tensor,
        edge_wigner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source, target = edge_index
        edge_feats = self.linear_up(
            torch.cat((node_feats[source], node_feats[target]), dim=-1)
        )
        weights = self.edge_info(edge_radial_basis)
        return self.rejector(edge_feats, edge_attrs, weights * edge_cutoff)


class O2EdgeInteraction(O3CgtpEdgeInteraction):
    """Apply radially weighted O(2) Linear--Gate--Linear in an edge-aligned frame."""

    def _setup_convolution(self) -> None:
        self.linear_up = e3nnLinear(self.irreps_in, self.irreps_hidden, bias=False)
        self.irreps_out = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_out]
        )
        self.rejector = UvO2TensorProduct(
            self.irreps_hidden + self.irreps_hidden, self.irreps_out
        )
        self.weight_numel = self.rejector.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        edge_cutoff: torch.Tensor,
        edge_wigner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_feats = self.linear_up(node_feats)
        source, target = edge_index
        edge_feats = torch.cat((node_feats[source], node_feats[target]), dim=-1)
        weights = self.edge_info(edge_radial_basis)
        return self.rejector(edge_feats, edge_wigner, weights * edge_cutoff)


EDGE_INTERACTION: dict[str, type[EdgeInteraction]] = {
    "cgtp": O3CgtpEdgeInteraction,
    "o2": O2EdgeInteraction,
}
