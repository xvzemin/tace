################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from tace.utils.torch_scatter import scatter_sum

from ..linear import IndexedFeatures
from ..mlp import MLP
from .base import ScatterNorm


class IdentityScatterNorm(ScatterNorm):
    """Leave aggregated features unchanged."""

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor | None,
        num_nodes: int,
    ) -> torch.Tensor:
        return node_feats


class AvgNumNeighborsScatterNorm(ScatterNorm):
    """Divide aggregated features by the mean neighbor count."""

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor | None,
        num_nodes: int,
    ) -> torch.Tensor:
        return node_feats / self.avg_num_neighbors


class SqrtAvgNumNeighborsScatterNorm(ScatterNorm):
    """Divide aggregated features by the square root of the mean neighbor count."""

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor | None,
        num_nodes: int,
    ) -> torch.Tensor:
        return node_feats / self.avg_num_neighbors**0.5


class DensityScatterNorm(ScatterNorm):
    r"""Divide features by a learned local density.

    The denominator is
    :math:`\alpha + \beta \sum_j c_{ij}\tanh(f(e_{ij})^2)`, where
    :math:`f` is an MLP with one hidden layer of width 64 and :math:`c_{ij}`
    is the edge cutoff. Initially, :math:`\alpha=\sqrt{\bar{N}}` and
    :math:`\beta=0`, with mean neighbor count :math:`\bar{N}`.
    """

    apply_cutoff = True

    def __init__(
        self,
        avg_num_neighbors: float,
        edge_feats_channel: int,
        radial_bias: bool = False,
        radial_layer_norm: bool = False,
    ) -> None:
        super().__init__(
            avg_num_neighbors, edge_feats_channel, radial_bias, radial_layer_norm
        )
        self.edge_density = MLP(
            [edge_feats_channel, 64, 1],
            bias=radial_bias,
            layer_norm=radial_layer_norm,
            act="silu",
        )
        self.alpha = torch.nn.Parameter(torch.tensor(avg_num_neighbors**0.5))
        self.beta = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor | None,
        num_nodes: int,
    ) -> torch.Tensor:
        density = torch.tanh(self.edge_density(edge_feats) ** 2)
        if edge_cutoff is not None and self.apply_cutoff:
            density = density * edge_cutoff
        density = scatter_sum(density, edge_index[1], dim=0, dim_size=num_nodes)
        density = density[: node_feats.size(0)]
        density = density * self.beta + self.alpha
        return node_feats / density


class NoCutoffDensityScatterNorm(DensityScatterNorm):
    """Normalize by learned local density without an edge cutoff factor."""

    apply_cutoff = False


SCATTER_NORM: dict[str | None, type[ScatterNorm]] = {
    None: IdentityScatterNorm,
    "identity": IdentityScatterNorm,
    "avg_num_neighbors": AvgNumNeighborsScatterNorm,
    "sqrt_avg_num_neighbors": SqrtAvgNumNeighborsScatterNorm,
    "density": DensityScatterNorm,
    "no_cutoff_density": NoCutoffDensityScatterNorm,
}
