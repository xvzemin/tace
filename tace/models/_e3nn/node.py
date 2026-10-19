################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import math
from typing import Union

import torch
from e3nn import o3

from eqx import o2

from ...utils.torch_scatter import scatter_sum
from ..layout import LayoutTransform
from ..linear import e3nnElementLinear, e3nnLinear
from ..mlp import MLP_TYPES
from .base import NodeEmbedding, NodeUpdate


class LinearNodeEmbedding(NodeEmbedding):
    """
    A simple node embedding module based on a linear transformation.

    This class projects discrete node attributes (e.g., element types)
    into a continuous feature space using a single linear layer,
    without introducing nonlinearity or structural information.
    """
    def _setup(self) -> None:
        self.embedding = torch.nn.Embedding(self.num_elements, self.num_channel)
        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "embedding.weight"
        legacy = prefix + "elem_emb1."
        weight_key = legacy + "weight"
        if weight_key not in state_dict:
            weight_key += ".0"
        if key not in state_dict and weight_key in state_dict:
            # Preserve the one-hot linear normalization in the lookup table.
            weight = state_dict.pop(weight_key).to(self.embedding.weight)
            weight = weight.reshape(self.num_elements, self.num_channel)
            weight = weight * self.num_elements**-0.5
            state_dict[key] = weight
            for name in ("linear.weight", "linear.bias", "linear.output_mask"):
                state_dict.pop(legacy + name, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        wigner,
        wigner_inv: Union[torch.Tensor, None],
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_type is None:
            node_type = node_attrs.argmax(dim=-1)
        return self.embedding(node_type)


class SphericalTensorNodeEmbedding(NodeEmbedding):
    """Aggregate element scalars weighted by radial functions and harmonics.

    Radial weights depend only on distance. The output contains natural-parity
    degrees up to ``min(Lmax, lmax)``, with ``num_channel`` copies per degree.

    Parameters
    ----------
    num_elements : int
        Number of chemical elements.
    num_radial_basis : int
        Input radial basis width.
    num_channel : int
        Channel multiplicity of each output irrep.
    Lmax, lmax : int
        Node and angular degree cutoffs.
    avg_num_neighbors : float
        Mean neighbor count used to normalize the sum.
    bias : bool, optional
        Include biases in scalar embedding and radial projections.
    radial_mlp : list of int, optional
        Radial MLP hidden widths. Defaults to two layers of ``num_channel``.
    radial_mlp_type : {"mlp", "mlp_v2"}, optional
        Radial network initialization and activation convention.
    radial_mlp_norm : {"layer_norm", "rms_norm"} or None, optional
        Hidden normalization. ``None`` preserves the width-dependent LayerNorm
        for ``mlp`` and disables normalization for ``mlp_v2``.
    """

    use_wigner = False
    element_dependent = False

    def _setup(self) -> None:
        self.irreps_out = o3.Irreps(
            [
                (self.num_channel, (ell, (-1) ** ell))
                for ell in range(min(self.Lmax, self.lmax) + 1)
            ]
        )
        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        input_dim = self.num_radial_basis + (
            2 * self.num_channel if self.element_dependent else 0
        )
        radial_norm = self.radial_mlp_norm
        if radial_norm is None:
            radial_norm = (
                "layer_norm"
                if self.radial_mlp_type == "mlp" and input_dim != self.num_radial_basis
                else None
            )
        self.edge_info = MLP_TYPES[self.radial_mlp_type](
            [
                input_dim,
                *self.radial_mlp,
                len(self.irreps_out) * self.num_channel,
            ],
            bias=self.bias,
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
        )
        self.normalization = math.sqrt(self.avg_num_neighbors)
        self.angular_sections = [slice(ir.l**2, (ir.l + 1)**2) for _, ir in self.irreps_out]
        self.reshape = LayoutTransform(
            self.irreps_out,
            layout_in="flatten_ir_mul",
            layout_out="flatten_mul_ir",
        )
        if self.use_wigner:
            self.frame = o2.LocalFrame(self.irreps_out, mmax=0, reverse=True)

    def forward(
        self,
        node_attrs,
        edge_feats,
        edge_index,
        edge_attrs,
        cutoff,
        wigner,
        wigner_inv,
        node_type=None,
    ):
        """Return aggregated node features in flattened ``mul_ir`` layout."""
        source, target = edge_index
        scalars = self.node_embedding(node_attrs)
        if self.element_dependent:
            edge_feats = torch.cat(
                (edge_feats, scalars[source], scalars[target]), dim=-1
            )
        weights = self.edge_info(edge_feats).view(
            source.shape[0], len(self.irreps_out), self.num_channel
        )
        features = scalars[source, None, :] * weights
        if self.use_wigner:
            message = self.frame.to_global(features.flatten(1), wigner_inv)
        else:
            message = torch.cat(
                [
                    (
                        edge_attrs[:, self.angular_sections[i], None]
                        * features[:, i, None, :]
                    ).flatten(1)
                    for i, (_, ir) in enumerate(self.irreps_out)
                ],
                dim=-1,
            )
        if cutoff is not None:
            message = message * cutoff
        node_feats = (
            scatter_sum(message, target, dim=0, dim_size=node_attrs.shape[0])
            / self.normalization
        )
        return self.reshape(node_feats)


class Element2SphericalTensorNodeEmbedding(SphericalTensorNodeEmbedding):
    """Tensor embedding with radial weights conditioned on both elements."""

    element_dependent = True


class WignerTensorNodeEmbedding(SphericalTensorNodeEmbedding):
    """Lift local scalars with inverse Wigner matrices and aggregate at nodes.

    The scalar lift includes component spherical-harmonic normalization.
    Radial weights depend only on distance.
    """

    use_wigner = True


class Element2WignerTensorNodeEmbedding(WignerTensorNodeEmbedding):
    """Wigner tensor embedding with weights conditioned on both elements."""

    element_dependent = True


class IdentityNodeUpdate(NodeUpdate):
    """Return the magnetic radial node information for both endpoints."""

    def _setup(self) -> None:
        self.out_dim = self.num_radial_basis

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return magnetic_radial_basis, magnetic_radial_basis


class ElementNodeUpdate(NodeUpdate):
    """Apply one shared element-dependent map to both endpoints."""

    def _setup(self) -> None:
        self.out_dim = self.num_channel
        self.embedding = e3nnElementLinear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            num_elements=self.num_elements,
            bias=self.use_bias,
        )

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_info = self.embedding(magnetic_radial_basis, node_attrs, node_type)
        return node_info, node_info


class Element2NodeUpdate(NodeUpdate):
    """Apply independent element-dependent maps to the two endpoints."""

    def _setup(self) -> None:
        self.out_dim = self.num_channel
        irreps_in = f"{self.num_radial_basis}x0e"
        irreps_out = f"{self.num_channel}x0e"
        self.source_embedding = e3nnElementLinear(
            irreps_in,
            irreps_out,
            num_elements=self.num_elements,
            bias=self.use_bias,
        )
        self.target_embedding = e3nnElementLinear(
            irreps_in,
            irreps_out,
            num_elements=self.num_elements,
            bias=self.use_bias,
        )

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.source_embedding(magnetic_radial_basis, node_attrs, node_type),
            self.target_embedding(magnetic_radial_basis, node_attrs, node_type),
        )


NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "spherical_tensor": SphericalTensorNodeEmbedding,
    "spherical_tensor_element2": Element2SphericalTensorNodeEmbedding,
    "wigner_tensor": WignerTensorNodeEmbedding,
    "wigner_tensor_element2": Element2WignerTensorNodeEmbedding,
}

NODE_UPDATE = {
    "identity": IdentityNodeUpdate,
    "element": ElementNodeUpdate,
    "element2": Element2NodeUpdate,
}
