################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
# Description: This file contains linear readout function for tensor rank > 0
################################################################################

from typing import List, Optional


import torch
from torch import nn, Tensor


from .linear import Linear
from .act import ACT
from .utils import expand_dims_to, select_corresponding_level_for_tensor


class NodeLinearReadOut(torch.nn.Module):
    """For tensor rank > 0"""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        atomic_numbers: List[int] = None,
    ) -> None:
        super().__init__()

        self.out = Linear(
            in_dim=in_dim,
            out_dim=out_dim,
            bias=bias,
            atomic_numbers=atomic_numbers,
        )

    def forward(
        self,
        TN: Tensor,
        T0: Optional[Tensor] = None,
        node_attrs: Optional[Tensor] = None,
        node_level: Optional[Tensor] = None
    ) -> Tensor:
        return self.out(TN, node_attrs)


class NodeNonLinearReadOut(torch.nn.Module):
    '''
    Using MLP to produce rank-0 and rank-0-gated together may be better or not, 
    But I'm too lazy to change it.
    '''
    def __init__(
        self,
        in_dim: int,
        hidden_dim: List[int],
        out_dim: int,
        bias: bool = False,
        atomic_numbers: List[int] = None,
        gate: str = "silu",
        num_levels: int = 1,
        enable_multi_head: bool = False,
    ) -> None:
        super().__init__()

        self.num_levels = num_levels
        self.enable_multi_head = (enable_multi_head) and (num_levels > 1) and len(hidden_dim) > 0
        if self.enable_multi_head:
            assert len(hidden_dim) == 1, 'For multihead training, cfg.model.config.readout_mlp.hidden_dim must be only one neuron'

        self.layer0s = nn.ModuleList()
        self.layerNs = nn.ModuleList()
        self.gate = ACT.get(gate, ACT['silu'])()

        prev_dim = in_dim
        for h_dim in hidden_dim:
            layer0 = Linear(
                in_dim=prev_dim,
                out_dim=h_dim,
                bias=bias,
                atomic_numbers=atomic_numbers,
            )
            layerN = Linear(
                in_dim=prev_dim,
                out_dim=h_dim,
                bias=bias,
                atomic_numbers=atomic_numbers,
            )
            self.layer0s.append(layer0)
            self.layerNs.append(layerN)
            prev_dim = h_dim

        self.output_layer = Linear(
            in_dim=prev_dim,
            out_dim=out_dim,
            bias=bias,
            atomic_numbers=atomic_numbers,
        )

    def forward(self, TN: Tensor, T0: Tensor,  node_attrs: Optional[Tensor] = None, node_level: Optional[Tensor] = None) -> Tensor:
        for linear0, linearN in zip(self.layer0s, self.layerNs):
            T0 = self.gate(linear0(T0, node_attrs))
            TN = expand_dims_to(T0, TN.ndim) * linearN(TN)
        if self.enable_multi_head:
            out = self.output_layer(select_corresponding_level_for_tensor(TN, node_level, self.num_levels))
        else:
            out = self.output_layer(TN)
        return out
