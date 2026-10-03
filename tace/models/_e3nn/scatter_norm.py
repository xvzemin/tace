# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################
# # TODO

# import abc
# from typing import Union


# import torch


# from tace.utils.torch_scatter import scatter_sum
# from ..mlp import FFN

# class DensityScatterNorm(torch.nn.Module):
#     def __init__(self, use_cutoff: bool = True) -> None:
#         super().__init__()

#     def forward(
#             self,
#             node_feats: torch.Tensor,
#             edge_feats: torch.Tensor,
#             edge_index: torch.Tensor,
#             cutoff: torch.Tensor,
#             alpha: torch.Tensor,
#             beta: torch.Tensor,
#             fn: torch.nn.Module,
#         ):

#             density = torch.tanh(fn(edge_feats) ** 2)
#             if cutoff is not None and self.apply_density_cutoff:
#                 density = density * cutoff
#             # density = density * cutoff
#             density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
#             density  = self.truncate_ghosts(density , nlocal)
#             density = density * self.beta + self.alpha

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}


# def get_scatter_norm_layer(
#         scatter_norm_type: str,
#         avg_num_neighbors: float,
#         edge_feats_channel: int,
#         edge_info_type,
#         radial_bias,
#         radial_layer_norm,
#     ):

#     if scatter_norm_type == 'density':  # From MACE
#         edge_density = FFN[edge_info_type](
#             [edge_feats_channel, 64, 1],
#             bias=radial_bias,
#             layer_norm=radial_layer_norm,
#             act='silu',
#         )
#         alpha = torch.nn.Parameter(torch.tensor(avg_num_neighbors))
#         beta = torch.nn.Parameter(torch.tensor(0.0))
#         return {
#             'edge_density': edge_density,
#             'alpha': alpha,
#             'beta': beta,
#         }

#     if scatter_norm_type == 'no_cutoff_density':
#         edge_density = FFN[edge_info_type](
#             [edge_feats_channel, 64, 1],
#             bias=radial_bias,
#             layer_norm=radial_layer_norm,
#             act='silu',
#         )
#         alpha = torch.nn.Parameter(torch.tensor(avg_num_neighbors))
#         beta = torch.nn.Parameter(torch.tensor(0.0))
#         return {
#             'edge_density': edge_density,
#             'alpha': alpha,
#             'beta': beta,
#         }
