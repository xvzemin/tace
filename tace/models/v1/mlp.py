################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import prod, sqrt
from typing import Optional, List


import torch
from torch import nn


from .act import ACT
from .utils import select_corresponding_level_for_scalar


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: List[int] = [],
        act: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
        enable_layer_norm: bool = False,
        num_levels: int = 1,
        enable_multi_head: bool = False,
    ):
        '''The parameter initialization method uses the earlier version of Allegro'''
        super().__init__()

        self.bias = bias
        self.dims = [in_dim] + hidden_dim + [out_dim]
        self.num_layers = len(self.dims) - 1
        assert self.num_layers >= 1
        self.bias = bias
        act = ACT[act]()
        self.is_nonlinear = False
        self.enable_layer_norm = enable_layer_norm
        self.num_levels = num_levels
        self.enable_multi_head = (enable_multi_head) and (num_levels > 1) and len(hidden_dim) > 0
        if self.enable_multi_head:
            assert len(hidden_dim) == 1, 'For multihead training, cfg.model.config.readout_mlp.hidden_dim must be only one neuron'

        # === build the MLP + weight init ===
        mlp = []
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):

            # === weight initialization ===
            if forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if act is None or (layer == 0) else sqrt(2)
            else:
                norm_dim = h_out
                gain = (
                    1.0 if act is None or (layer == self.num_layers - 1) else sqrt(2)
                )

            # === instantiate Linear ===
            linear_layer = LinearLayer(
                in_dim=h_in,
                out_dim=h_out,
                alpha=gain / sqrt(norm_dim),
                bias=bias,
            )
            mlp.append(linear_layer)

            # === optional LayerNorm ===
            if enable_layer_norm:
                if layer < len(self.dims) -2:
                    mlp.append(nn.LayerNorm(h_out))
            del gain, norm_dim

            # === act ===
            if (layer != self.num_layers - 1) and (act is not None):
                mlp.append(act)
                self.is_nonlinear = True

        if (not self.is_nonlinear) and (not self.bias) and (self.num_layers > 1):
            self.mlp = DeepLinearLayer(torch.nn.Sequential(*mlp))
            del mlp
        elif self.enable_multi_head: 
            self.mlp_1 = torch.nn.Sequential(*mlp[:-1]) 
            self.mlp_2 = torch.nn.Sequential(mlp[-1])   
        else:
            self.mlp = torch.nn.Sequential(*mlp)


    def forward(self, x, node_level=None):
        if self.enable_multi_head:
            x = self.mlp_1(x)
            x = select_corresponding_level_for_scalar(x, node_level, self.num_levels)
            return self.mlp_2(x)
        else:
            return self.mlp(x)


class DeepLinearLayer(torch.nn.Module):
    def __init__(self, mlp) -> None:

        super().__init__()
        self.weights = torch.nn.ParameterList()
        alphas = []
        for this_idx, mlp_idx in enumerate(range(len(mlp))):
            new_weight = torch.clone(mlp[mlp_idx].weight)
            self.weights.append(new_weight)
            del new_weight
            alphas.append(mlp[mlp_idx].alpha)
        self.alpha = prod(alphas)
        del alphas

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.mul(
            torch.linalg.multi_dot([weight for weight in self.weights]), self.alpha
        )
        return torch.mm(input, weight)

    
class LinearLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        alpha: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        self._bias = bias
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.alpha
        if self.bias is None:
            return torch.mm(input, weight)
        else:
            return torch.addmm(self.bias, input, weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim} bias={ self._bias}, alpha={self.alpha:.2f})"