################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch

from eqx import co3

from .base import EdgeEmbedding, EdgeUpdate
from .linear import Linear


class IdentityEdgeEmbedding(EdgeEmbedding):
    def _setup(self) -> None:
        self.out_dim = self.num_radial_basis

    def forward(self, node_feats, node_attrs, edge_feats, edge_index, cutoff):
        return edge_feats


class LinearEdgeEmbedding(EdgeEmbedding):
    def _setup(self) -> None:
        self.out_dim = self.num_channel
        self.radial_proj = Linear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

    def forward(self, node_feats, node_attrs, edge_feats, edge_index, cutoff):
        return self.radial_proj(edge_feats)


class NonLinearEdgeEmbedding(LinearEdgeEmbedding):
    def _setup(self) -> None:
        super()._setup()
        self.act = co3.Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])

    def forward(self, node_feats, node_attrs, edge_feats, edge_index, cutoff):
        return self.act(self.radial_proj(edge_feats))


class IdentityEdgeUpdate(EdgeUpdate):
    def _setup(self) -> None:
        self.out_dim = self.edge_embedding_channel

    def forward(self, node_feats, node_attrs, edge_feats, edge_index, cutoff):
        return edge_feats


class ElementEdgeUpdate(EdgeUpdate):
    def _setup(self) -> None:
        self.out_dim = self.edge_embedding_channel + 2 * self.num_channel
        self.source_embedding = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.use_bias,
        )
        self.target_embedding = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.use_bias,
        )
        torch.nn.init.uniform_(self.source_embedding.weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, -0.001, 0.001)

    def _element_features(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.source_embedding(node_attrs[edge_index[0]])
        target = self.target_embedding(node_attrs[edge_index[1]])
        return source, target

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source, target = self._element_features(node_attrs, edge_index)
        return torch.cat((edge_feats, source, target), dim=-1)


class Element2EdgeUpdate(ElementEdgeUpdate):
    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source, target = self._element_features(node_attrs, edge_index)
        return torch.cat((edge_feats, target, source), dim=-1)


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element": ElementEdgeUpdate,
    "element2": Element2EdgeUpdate,
}


__all__ = ["EDGE_EMBEDDING", "EDGE_UPDATE"]
