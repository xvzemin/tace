################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections import Counter
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
        node_type: torch.Tensor,
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
        node_type: torch.Tensor,
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
        node_type: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return self.act1(self.radial_proj(edge_feats))


class ElementEdgeEmbedding(IdentityEdgeEmbedding):
    """Share one element offset per CGTP path and multiplicity channel.

    The radial features are unchanged. Both endpoints use the same table.
    Its rows are initialized to zero. In each layer the mean of the endpoint
    rows, indexed by that layer's instructions, is added to the radial weights
    before applying the cutoff.
    """

    table_names = ("embedding",)

    @staticmethod
    def path_keys(conv):
        """Return semantic keys in the convolution's external weight order."""

        def channels(irreps):
            counts = Counter()
            blocks = []
            for mul, ir in irreps:
                key = (ir.l, ir.p, getattr(ir, "t", 1))
                start = counts[key]
                blocks.append((key, range(start, start + mul)))
                counts[key] += mul
            return blocks

        node, angular = channels(conv.irreps_in1), channels(conv.irreps_in2)
        copies = Counter()
        keys = []
        for instruction in conv.instructions:
            i, j, k, mode, weighted = instruction[:5]
            if not weighted:
                continue
            if mode != "uvu":
                raise ValueError("Element edge embedding requires uvu CGTP paths.")
            ir = conv.irreps_out[k].ir
            out = (ir.l, ir.p, getattr(ir, "t", 1))
            a, us = node[i]
            b, vs = angular[j]
            copy = copies[i, j, out]
            copies[i, j, out] += 1
            keys.extend((out, a, b, copy, u, v) for u in us for v in vs)
        if len(keys) != conv.weight_numel:
            raise ValueError("CGTP paths do not match the convolution weight layout.")
        return keys

    def build_paths(self, convolutions) -> None:
        """Index each layer into the union of its weighted CGTP paths."""
        layers = [self.path_keys(conv) for conv in convolutions]
        keys = sorted(set().union(*layers))
        columns = {key: i for i, key in enumerate(keys)}
        for name in self.table_names:
            setattr(
                self,
                name,
                torch.nn.Embedding.from_pretrained(
                    torch.zeros(self.num_elements, len(keys)), freeze=False
                ),
            )
        for layer, paths in enumerate(layers):
            index = [columns[key] for key in paths]
            self.register_buffer(
                f"path_index_{layer}",
                None if paths == keys else torch.tensor(index, dtype=torch.long),
                persistent=False,
            )

    def layer_weights(self, layer: int) -> torch.Tensor:
        """Return the element table in the requested layer's path order."""
        index = getattr(self, f"path_index_{layer}")
        table = self.embedding.weight
        return table if index is None else table.index_select(1, index)


class Element2EdgeEmbedding(ElementEdgeEmbedding):
    """Add the mean of independent source and target element rows to radial path weights.

    Each bias-free table has one row per element and one column per distinct
    weighted path and channel. Both are zero-initialized and shared across layers.
    """

    table_names = ("source_embedding", "target_embedding")

    def layer_weights(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = getattr(self, f"path_index_{layer}")
        source, target = self.source_embedding.weight, self.target_embedding.weight
        if index is None:
            return source, target
        return source.index_select(1, index), target.index_select(1, index)


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
        node_type: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return edge_feats


class Element2EdgeUpdate(EdgeUpdate):
    """Return radial, target and source features for the edge MLP.

    Element lookup tables are evaluated once per node and carry gather indices.
    The first MLP linear map projects each input before gathering and adding
    the results, without materializing concatenated edge features.
    """

    def _setup(self) -> None:

        self.out_dim = self.edge_embedding_channel + self.num_channel * 2

        self.source_embedding = torch.nn.Embedding(self.num_elements, self.num_channel)
        self.target_embedding = torch.nn.Embedding(self.num_elements, self.num_channel)
        bound = 0.001 * self.num_elements**-0.5
        torch.nn.init.uniform_(self.source_embedding.weight, -bound, bound)
        torch.nn.init.uniform_(self.target_embedding.weight, -bound, bound)

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
        for name in ("source_embedding", "target_embedding"):
            embedding = getattr(self, name)
            key = prefix + name + "."
            weight_key = key + "weight"
            legacy_key = weight_key + ".0"
            if legacy_key not in state_dict:
                legacy_key = weight_key
            weight = state_dict.get(legacy_key)
            if weight is not None and (
                legacy_key != weight_key
                or weight.ndim == 1
                or key + "linear.output_mask" in state_dict
            ):
                # Fold the one-hot linear normalization and bias into each row.
                weight = state_dict.pop(legacy_key).to(embedding.weight)
                weight = weight.reshape(self.num_elements, self.num_channel)
                weight = weight * self.num_elements**-0.5
                bias = state_dict.pop(key + "bias", None)
                if bias is not None:
                    weight = weight + bias.to(weight)
                state_dict[weight_key] = weight
                for field in ("linear.weight", "linear.bias", "linear.output_mask"):
                    state_dict.pop(key + field, None)
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
        node_feats: torch.Tensor,
        node_type: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
    ) -> IndexedFeatures:
        return (
            (edge_feats, None),
            (self.target_embedding(node_type), edge_index[1]),
            (self.source_embedding(node_type), edge_index[0]),
        )


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "element": ElementEdgeEmbedding,
    "element2": Element2EdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element2": Element2EdgeUpdate,
}
