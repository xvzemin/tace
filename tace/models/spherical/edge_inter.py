################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

from ..linear import e3nnLinear
from ..mlp import BATCH_MLP, MLP
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
        options = dict(
            bias=radial_basis["bias"],
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
        )
        widths = radial_basis.get("widths")
        if widths is None:
            self.edge_info = MLP[self.mlp_type](
                [radial_basis["num_radial_basis"] + 2 * self.num_elements]
                + list(radial_basis["hidden"])
                + [self.weight_numel],
                **options,
            )
        else:
            self.edge_info = BATCH_MLP[self.mlp_type].from_product(
                self.rejector,
                widths,
                radial_basis["hidden"],
                context_features=2 * self.num_elements,
                **options,
            )
            self.register_buffer(
                "element_basis", torch.eye(self.num_elements), persistent=False
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
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source, target = edge_index
        edge_feats = self.linear_up(
            torch.cat((node_feats[source], node_feats[target]), dim=-1)
        )
        weights = self.radial_weights(edge_radial_basis, node_type, edge_index)
        return self.rejector(edge_feats, edge_attrs, weights * edge_cutoff)

    def radial_weights(self, radial, node_type, edge_index):
        """Project radial features and element-table rows without one-hot inputs."""
        if self.radial_basis.get("widths") is not None:
            source, target = edge_index
            return self.edge_info(
                (
                    (radial, None),
                    (self.element_basis, node_type[source]),
                    (self.element_basis, node_type[target]),
                )
            )
        linear = self.edge_info.mlp[0]
        weight = linear.get_weight()
        radial_width = self.radial_basis["num_radial_basis"]
        source, target = edge_index
        radial_weight, source_weight, target_weight = weight.split(
            (radial_width, self.num_elements, self.num_elements), dim=0
        )
        output = radial @ radial_weight
        output = (
            output
            + source_weight[node_type[source]]
            + target_weight[node_type[target]]
        )
        if linear.bias is not None:
            output = output + linear.bias
        return self.edge_info.mlp[1:](output)


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
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_feats = self.linear_up(node_feats)
        source, target = edge_index
        edge_feats = torch.cat((node_feats[source], node_feats[target]), dim=-1)
        weights = self.radial_weights(edge_radial_basis, node_type, edge_index)
        return self.rejector(edge_feats, edge_wigner, weights * edge_cutoff)


EDGE_INTERACTION: dict[str, type[EdgeInteraction]] = {
    "cgtp": O3CgtpEdgeInteraction,
    "o2": O2EdgeInteraction,
}
