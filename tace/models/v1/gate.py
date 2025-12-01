################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'GATE not use in TACE now'
import torch
from torch import nn, Tensor


from .utils import expand_dims_to


class TensorSilu(torch.nn.Module):
    """
    This gate function is modified from HotPP.
    silu(x) = x * sigmoid(x)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(in_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(in_dim, requires_grad=True))
        self.in_dim = in_dim

    def forward(self, t: Tensor, r: int) -> Tensor:
        B = t.size(0)
        C = t.size(1)
        t_ = t.reshape(B, C, -1)
        norm = self.weights * torch.sum(t_ ** 2, dim=2) + self.bias
        factor = torch.sigmoid(norm)
        factor = expand_dims_to(factor, 2 + r)
        return  factor * t

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_dim={self.in_dim}, "
                f"weights_shape={tuple(self.weights.shape)}, "
                f"bias_shape={tuple(self.bias.shape)})")


class TensorIdentity(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, t: Tensor, r: int) -> Tensor:
        return t


    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.in_dim})"
    
GATE = {
    "identity": TensorIdentity,
    "silu": TensorSilu,
}

