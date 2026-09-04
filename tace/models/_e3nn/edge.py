################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch
from e3nn.nn import Activation

from ..linear import e3nnElementLinear, e3nnLinear
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


# class ElementEdgeEmbedding(EdgeEmbedding):
#     """
#     An edge embedding module that incorporates both radial and element information.

#     This class combines transformed edge features with embeddings of the source
#     and target nodes, allowing the edge representation to depend not only on
#     geometric information but also on the types of connected elements.

#     Note:
#         When using this module, it is recommended not to additionally use
#         edge update modules that rely purely on element information, as this
#         may lead to an overemphasis on element features in the edge representation.
#     """

#     def _setup(self) -> None:

#         self.out_dim = self.num_channel * 3

#         self.radial_proj = e3nnLinear(
#             f"{self.num_radial_basis}x0e",
#             f"{self.num_channel}x0e",
#             bias=self.bias,
#         )
#         self.source_embedding = e3nnLinear(
#             f"{self.num_elements}x0e",
#             f"{self.num_channel}x0e",
#             bias=self.bias,
#         )
#         self.target_embedding = e3nnLinear(
#             f"{self.num_elements}x0e",
#             f"{self.num_channel}x0e",
#             bias=self.bias,
#         )
#         self.act1 = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])

#         if isinstance(self.source_embedding.weight, torch.Tensor):
#             torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
#             torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)
#         else:
#             torch.nn.init.uniform_(self.source_embedding.weight[0], a=-0.001, b=0.001)
#             torch.nn.init.uniform_(self.target_embedding.weight[0], a=-0.001, b=0.001)

#     def forward(
#         self,
#         node_feats: torch.Tensor,
#         node_attrs: torch.Tensor,
#         edge_feats: torch.Tensor,
#         edge_index: torch.Tensor,
#         cutoff: Union[torch.Tensor, None],
#     ) -> torch.Tensor:

#         assert cutoff is not None, "Please set radial_basis.apply_cutoff = False"

#         x_j = self.source_embedding(node_attrs)[edge_index[0]]
#         x_i = self.target_embedding(node_attrs)[edge_index[1]]
#         edge_feats = self.radial_proj(edge_feats)

#         return self.act1(torch.cat([edge_feats, x_i, x_j], dim=-1))


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


class ElementEdgeUpdate(EdgeUpdate):
    """
    An edge update module that incorporates edge element information.

    This class augments edge features by concatenating embeddings of the
    source and target node elements, allowing edge representations to
    explicitly depend on the types of connected nodes.
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
    ) -> torch.Tensor:

        edge_feats_list = [edge_feats]
        edge_feats_list.append(self.source_embedding(node_attrs[edge_index[0]]))
        edge_feats_list.append(self.target_embedding(node_attrs[edge_index[1]]))
        return torch.cat(edge_feats_list, dim=-1)


class Element2EdgeUpdate(ElementEdgeUpdate):
    """
    A variant of element-based edge update with reversed ordering.

    This class is similar to ElementEdgeUpdate but swaps the order of
    source and target element embeddings when concatenating.
    """

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        edge_feats_list = [edge_feats]
        edge_feats_list.append(self.target_embedding(node_attrs[edge_index[1]]))
        edge_feats_list.append(self.source_embedding(node_attrs[edge_index[0]]))
        return torch.cat(edge_feats_list, dim=-1)


class IdentityMagneticEdgeUpdate(torch.nn.Module):
    """Gather node magnetic radial bases without transforming them."""

    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.use_bias = bias
        self.out_dim = 2 * num_radial_basis

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        return torch.cat(
            (magnetic_radial_basis[source], magnetic_radial_basis[target]),
            dim=-1,
        )


class MagneticEdgeUpdate(torch.nn.Module):
    """Build magnetic edge features using one shared element linear."""

    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.out_dim = 2 * num_channel

        irreps_in = f"{num_radial_basis}x0e"
        irreps_out = f"{num_channel}x0e"
        self.embedding = e3nnElementLinear(
            irreps_in,
            irreps_out,
            num_elements=num_elements,
            bias=bias,
        )

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        node_features = self.embedding(magnetic_radial_basis, node_attrs)
        return torch.cat(
            (node_features[source], node_features[target]),
            dim=-1,
        )


class Magnetic2EdgeUpdate(torch.nn.Module):
    """Build magnetic edge features using separate endpoint element linears."""

    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.out_dim = 2 * num_channel

        irreps_in = f"{num_radial_basis}x0e"
        irreps_out = f"{num_channel}x0e"
        self.source_embedding = e3nnElementLinear(
            irreps_in,
            irreps_out,
            num_elements=num_elements,
            bias=bias,
        )
        self.target_embedding = e3nnElementLinear(
            irreps_in,
            irreps_out,
            num_elements=num_elements,
            bias=bias,
        )

    def forward(
        self,
        magnetic_radial_basis: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        source_features = self.source_embedding(magnetic_radial_basis, node_attrs)
        target_features = self.target_embedding(magnetic_radial_basis, node_attrs)
        return torch.cat(
            (source_features[source], target_features[target]),
            dim=-1,
        )


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
    # "element": ElementEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    # "element": ElementEdgeUpdate,
    "element2": Element2EdgeUpdate,
}

MAGNETIC_EDGE_UPDATE = {
    "identity": IdentityMagneticEdgeUpdate,
    "element": MagneticEdgeUpdate,
    "element2": Magnetic2EdgeUpdate,
}
