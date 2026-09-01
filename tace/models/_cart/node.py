################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch

from eqx import co3

from ..mlp import MLP
from .base import NodeEmbedding, natural_irreps
from .fused import O3ScatterTensorProduct
from .linear import Linear


class LinearNodeEmbedding(NodeEmbedding):
    def _setup(self) -> None:
        self.irreps_out = co3.Irreps(f"{self.num_channel}x0e")
        self.element_embedding = Linear(
            f"{self.num_elements}x0e",
            self.irreps_out,
            bias=self.bias,
        )

    def forward(
        self,
        node_attrs,
        edge_feats,
        edge_index,
        edge_attrs,
        cutoff,
        wigner=None,
        wigner_inv=None,
    ):
        return self.element_embedding(node_attrs)

    def __repr__(self) -> str:
        return repr(self.element_embedding)


class NonLinearNodeEmbedding(LinearNodeEmbedding):
    def _setup(self) -> None:
        super()._setup()
        self.activation = co3.Activation(
            self.irreps_out,
            [torch.nn.SiLU()],
        )

    def forward(self, *args, **kwargs):
        return self.activation(super().forward(*args, **kwargs))


class TensorNodeEmbedding(NodeEmbedding):
    def _setup(self) -> None:
        scalar_irreps = co3.Irreps(f"{self.num_channel}x0e")
        self.node_embedding = Linear(
            f"{self.num_elements}x0e", scalar_irreps, bias=self.bias
        )
        self.source_embedding = Linear(
            f"{self.num_elements}x0e", scalar_irreps, bias=self.bias
        )
        self.target_embedding = Linear(
            f"{self.num_elements}x0e", scalar_irreps, bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, -0.001, 0.001)
        self.irreps_edge = natural_irreps(self.lmax)
        self.irreps_out = natural_irreps(self.Lmax, self.num_channel)
        self.rejector = O3ScatterTensorProduct(
            scalar_irreps,
            self.irreps_edge,
            self.irreps_out,
        )
        self.edge_info = MLP(
            [
                self.num_radial_basis + 2 * self.num_channel,
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
        cutoff: Optional[torch.Tensor],
        wigner=None,
        wigner_inv=None,
    ) -> torch.Tensor:
        base = self.node_embedding(node_attrs)
        source = self.source_embedding(node_attrs[edge_index[0]])
        target = self.target_embedding(node_attrs[edge_index[1]])
        conv_weights = self.edge_info(torch.cat((edge_feats, source, target), dim=-1))
        if cutoff is not None:
            conv_weights = conv_weights * cutoff
        node_feats = (
            self.rejector(
                torch.ones_like(base),
                edge_attrs,
                conv_weights,
                edge_index,
            )
            / self.avg_num_neighbors
        )
        scalar_slice = self.irreps_out.slices()[0]
        node_feats = node_feats.clone()
        node_feats[..., scalar_slice] = node_feats[..., scalar_slice] + base
        return node_feats


NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonLinearNodeEmbedding,
    "tensor": TensorNodeEmbedding,
}


__all__ = ["NODE_EMBEDDING"]
