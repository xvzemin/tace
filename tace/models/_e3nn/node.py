################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import math
from typing import Union

import torch
from e3nn import o3

from ..linear import e3nnElementLinear, e3nnLinear
from ..mlp import MLP, get_scaled_activation
from .base import NodeEmbedding, NodeUpdate
from .fused import O3ScatterTensorProduct
from .o2 import O2ScatterTensorProduct


class LinearNodeEmbedding(NodeEmbedding):
    """
    A simple node embedding module based on a linear transformation.

    This class projects discrete node attributes (e.g., element types)
    into a continuous feature space using a single linear layer,
    without introducing nonlinearity or structural information.
    """

    def _setup(self) -> None:

        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

        self.elem_emb1 = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
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
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        return self.elem_emb1(node_attrs)

    def __repr__(self) -> str:
        return repr(self.elem_emb1)


class LinearSpinNodeEmbedding(NodeEmbedding):
    """Linear embed element types + magnetic-moment radial bases."""

    def _setup(self) -> None:
        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")
        self.element_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            self.irreps_out,
            bias=self.bias,
        )
        self.spin_embedding = e3nnLinear(
            f"{self.num_mag_radial_basis}x0e",
            self.irreps_out,
            bias=self.bias,
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
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if magnetic_radial_basis is None:
            raise ValueError("LinearSpinNodeEmbedding requires magnetic radial bases.")
        return (
            self.element_embedding(node_attrs)
            + self.spin_embedding(magnetic_radial_basis)
        ) / math.sqrt(2.0)


class NonLinearSpinNodeEmbedding(LinearSpinNodeEmbedding):
    """Embed element and spin inputs, followed by a scaled SiLU."""

    def _setup(self) -> None:
        super()._setup()
        self.activation = get_scaled_activation("silu")

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        wigner,
        wigner_inv: Union[torch.Tensor, None],
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        return self.activation(
            super().forward(
                node_attrs,
                edge_feats,
                edge_index,
                edge_attrs,
                cutoff,
                wigner,
                wigner_inv,
                magnetic_radial_basis,
            )
        )


class TensorNodeEmbedding(NodeEmbedding):
    def _setup(self) -> None:

        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

        self.rejector = O3ScatterTensorProduct(
            [(self.num_channel, (0, 1))],
            [(1, (l, (-1) ** l)) for l in range(self.lmax + 1)],
            [(1, (l, (-1) ** l)) for l in range(self.Lmax + 1)],
        )

        self.irreps_out = self.rejector.irreps_out

        self.edge_info = MLP(
            channels=[
                self.num_radial_basis + self.num_channel * 2,
                self.num_channel,
                self.num_channel,
                self.rejector.weight_numel,
            ],
            bias=True,
            layer_norm=True,
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
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        base_node_feats = self.node_embedding(node_attrs)
        source_feats = self.source_embedding(node_attrs[edge_index[0]])
        target_feats = self.target_embedding(node_attrs[edge_index[1]])
        conv_weights = self.edge_info(
            torch.cat([edge_feats, source_feats, target_feats], dim=-1)
        )
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        node_feats = (
            self.rejector(
                torch.ones_like(base_node_feats),
                edge_attrs,
                conv_weights,
                edge_index,
            )
            / self.avg_num_neighbors
        )

        node_feats[:, : self.num_channel] = (
            node_feats.narrow(1, 0, self.num_channel) + base_node_feats
        )

        return node_feats


class O2TensorNodeEmbedding(NodeEmbedding):
    def _setup(self) -> None:
        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)
        self.irreps_out = o3.Irreps(
            [(self.num_channel, (l, (-1) ** l)) for l in range(self.Lmax + 1)]
        )
        self.rejector = O2ScatterTensorProduct(
            self.node_embedding.irreps_out,
            self.irreps_out,
            num_channel=self.num_channel,
            lmax=max(self.Lmax, self.lmax),
            mmax=0,
            even_scalar_act=torch.nn.SiLU(),
            odd_scalar_act=torch.nn.Tanh(),
            tensor_act=torch.nn.Sigmoid(),
            num_head=1,
            num_radial_basis=self.num_radial_basis,
            use_radial_rotary_attention=False,
        )
        self.edge_info = MLP(
            channels=[
                self.num_radial_basis + self.num_channel * 2,
                self.num_channel,
                self.num_channel,
                self.rejector.weight_numel,
            ],
            bias=True,
            layer_norm=True,
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
        magnetic_radial_basis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        base_node_feats = self.node_embedding(node_attrs)
        source_feats = self.source_embedding(node_attrs[edge_index[0]])
        target_feats = self.target_embedding(node_attrs[edge_index[1]])
        conv_weights = self.edge_info(
            torch.cat([edge_feats, source_feats, target_feats], dim=-1)
        )
        node_feats = (
            self.rejector(
                torch.ones_like(base_node_feats),
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_cutoff=(
                    cutoff
                    if cutoff is not None
                    else edge_feats.new_ones(edge_feats.size(0), 1)
                ),
            )
            / self.avg_num_neighbors
        )
        node_feats[:, : self.num_channel] = (
            node_feats.narrow(1, 0, self.num_channel) + base_node_feats
        )
        return node_feats


class IdentityNodeUpdate(NodeUpdate):
    """Return the magnetic radial node features for both endpoints."""

    def _setup(self) -> None:
        self.out_dim = self.num_radial_basis

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = self.embedding(magnetic_radial_basis, node_attrs)
        return node_features, node_features


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.source_embedding(magnetic_radial_basis, node_attrs),
            self.target_embedding(magnetic_radial_basis, node_attrs),
        )


NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "linear_spin": LinearSpinNodeEmbedding,
    "nonlinear_spin": NonLinearSpinNodeEmbedding,
    "tensor": TensorNodeEmbedding,
    "o2_tensor": O2TensorNodeEmbedding,
}

NODE_UPDATE = {
    "identity": IdentityNodeUpdate,
    "element": ElementNodeUpdate,
    "element2": Element2NodeUpdate,
}
