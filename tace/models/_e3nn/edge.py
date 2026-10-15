################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Union

import torch
from e3nn.nn import Activation

from ..linear import IndexedFeatures, e3nnLinear
from .base import EdgeEmbedding, EdgeUpdate


class IdentityEdgeEmbedding(EdgeEmbedding):
    """
    An identity edge embedding module.

    This class directly returns the input edge features (radial) without any transformation.
    """

    def _setup(self) -> None:

        self.out_dim = self.num_radial_basis

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return edge_feats


class LinearEdgeEmbedding(EdgeEmbedding):
    """
    A linear edge embedding module.

    This class projects the input edge features (radial)
    into a higher-dimensional feature space using a linear transformation.

    This is motivated by the fact that when edge update are used,
    a low-dimensional radial representation may become a bottleneck and limit
    the expressiveness of edge features.
    """

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.radial_proj = e3nnLinear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return self.radial_proj(edge_feats)


class NonLinearEdgeEmbedding(EdgeEmbedding):
    """
    A nonlinear edge embedding module.

    This class applies a nonlinear activation function after a linear projection
    of edge features, allowing for more expressive representations compared to
    purely linear transformations.

    This is motivated by the fact that when edge update are used,
    a low-dimensional radial representation may become a bottleneck and limit
    the expressiveness of edge features.
    """

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.radial_proj = e3nnLinear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

        self.act1 = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return self.act1(self.radial_proj(edge_feats))


class IdentityEdgeUpdate(EdgeUpdate):
    """
    An identity edge update module.

    This class directly returns the input edge features (edge embedding) without modification.
    """

    def _setup(self) -> None:

        self.out_dim = self.edge_embedding_channel

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return edge_feats


class Element2EdgeUpdate(EdgeUpdate):
    """Return radial, target and source features for the edge MLP.

    Node embeddings retain their node layout and carry gather indices.
    The first MLP linear map projects each input before gathering and adding
    the results, without materializing concatenated edge features.
    """

    def _setup(self) -> None:

        self.out_dim = self.edge_embedding_channel + self.num_channel * 2

        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.use_bias,
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.use_bias,
        )
        if isinstance(self.source_embedding.weight, torch.Tensor):
            torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
            torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)
        else:
            torch.nn.init.uniform_(self.source_embedding.weight[0], a=-0.001, b=0.001)
            torch.nn.init.uniform_(self.target_embedding.weight[0], a=-0.001, b=0.001)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> IndexedFeatures:
        return (
            (edge_feats, None),
            (self.target_embedding(node_attrs), edge_index[1]),
            (self.source_embedding(node_attrs), edge_index[0]),
        )


class Element2EdgeUpdateV2(EdgeUpdate):
    r"""Prepare radial and endpoint features with branch-wise scaling.

    A branch of width :math:`d` is scaled by :math:`\sqrt{D/d}`, where
    :math:`D` is ``out_dim``. A randomly initialized unit-fan-in linear map
    preserves the radial second moment in expectation for zero embeddings.
    Element embeddings are initialized independently with ``ELEMENT_STD``.
    """

    ELEMENT_STD = 1e-3

    def _setup(self) -> None:
        self.out_dim = self.edge_embedding_channel + 2 * self.num_channel
        self.radial_scale = sqrt(self.out_dim / self.edge_embedding_channel)
        self.element_scale = sqrt(self.out_dim / self.num_channel)
        self.source_embedding = torch.nn.Embedding(self.num_elements, self.num_channel)
        self.target_embedding = torch.nn.Embedding(self.num_elements, self.num_channel)
        torch.nn.init.normal_(self.source_embedding.weight, std=self.ELEMENT_STD)
        torch.nn.init.normal_(self.target_embedding.weight, std=self.ELEMENT_STD)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        node_type: torch.Tensor | None = None,
    ) -> IndexedFeatures:

        if node_type is None:
            node_type = node_attrs.argmax(dim=-1)
        return (
            (edge_feats * self.radial_scale, None),
            (self.target_embedding(node_type) * self.element_scale, edge_index[1]),
            (self.source_embedding(node_type) * self.element_scale, edge_index[0]),
        )


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element2": Element2EdgeUpdate,
}
