################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Dict, List, Optional


import torch
from torch import Tensor, nn


from .utils import expand_dims_to


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        atomic_numbers: List[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = 1.0 / sqrt(in_dim)
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, T: Tensor, node_attrs: Optional[Tensor] = None) -> Tensor:
        B = T.size(0)
        C = T.size(1)
        REST = T.size()[2:]
        if B == 0:
            return T.new_empty((B, self.out_dim) + REST) # for isolated atoms 
        T = T.contiguous().reshape(B, C, -1).transpose(1, 2)
        T = T @ self.weight * self.alpha
        if self.bias is not None:
            T = T + self.bias
        T = T.transpose(1, 2).reshape((B, -1) + REST)
        return T

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias is not None})"


class ElementLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        atomic_numbers: List[int] = None,
    ):
        super().__init__()
        num_elemnts = len(atomic_numbers)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.atomic_numbers = atomic_numbers
        self.alpha = 1.0 / sqrt(in_dim)
        self.register_buffer(
            "num_elemnts", torch.tensor(num_elemnts, dtype=torch.int64)
        )
        self.weights = nn.Parameter(torch.empty(num_elemnts, out_dim, in_dim))
        torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_elemnts, out_dim))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, T: Tensor, node_attrs: Tensor) -> Tensor:
        B = T.size(0)
        C = T.size(1)
        REST = T.size()[2:]
        if B == 0:
            return T.new_empty((B, self.out_dim) + REST)  # for isolated atoms 
        idx = node_attrs.argmax(dim=-1)
        T = T.reshape(B, C, -1)
        W = self.weights[idx] * self.alpha
        T = torch.bmm(W, T)
        if self.bias is not None:
            b = self.bias[idx].unsqueeze(-1)
            T = T + b
        T = T.reshape((B, -1) + REST)
        return T

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias is not None})"


class SelfInteraction(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        rs: List[int],
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.rs = rs
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.linears = nn.ModuleDict(
            {
                str(r): Linear(
                    in_channel,
                    out_channel,
                    bias=(r == 0 and bias),
                )
                for r in rs
            }
        )

    def forward(self, ins: Dict[int, Tensor]) -> Dict[int, Tensor]:
        outs = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for r, linear in self.linears.items():
            r = int(r)
            outs[r] = linear(ins[r])
        return outs

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channel={self.in_channel}, out_channel={self.out_channel}, rank={self.rs})"

# only for prod
class CWLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        atomic_numbers: List[int] = None,
    ):
        super().__init__()
        self.channel = out_dim
        num_path = int(in_dim / out_dim)
        self.num_path = num_path
        self.atomic_numbers = atomic_numbers
        self.alpha = 1.0 / sqrt(num_path)
        self.weight = nn.Parameter(torch.empty(out_dim, num_path))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, node_attrs=None) -> Tensor:
        # (P, B, C, ...) => (B, C, P, ...)

        ndim = x.ndim
        device = x.device
        REST: List[int] = torch.arange(3, ndim, device=device).tolist()
        PERMUTE = [1, 2, 0] + REST
        x = x.permute(PERMUTE)

        W = self.weight.unsqueeze(0)
        W = expand_dims_to(W, n_dim=ndim, dim=-1)  # (1, C, P)

        # (B, C, 3, 3, ...)
        out = (x * W).sum(dim=2) * self.alpha

        if self.bias is not None:
            b = self.bias
            b = expand_dims_to(b, n_dim=out.ndim, dim=-1)
            out = out + b.view(-1, self.channel)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.num_path}, out_dim={self.channel}, bias={self.bias is not None})"

# only for prod
class ElementCWLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        atomic_numbers: List[int] = None,
    ):
        super().__init__()
        num_elements = len(atomic_numbers)
        self.channel = out_dim
        num_path = int(in_dim / out_dim)
        self.num_path = num_path
        self.atomic_numbers = atomic_numbers
        self.alpha = 1.0 / sqrt(num_path)

        self.register_buffer(
            "num_elements", torch.tensor(num_elements, dtype=torch.int64)
        )

        self.weights = nn.Parameter(torch.empty(num_elements, out_dim, num_path))
        torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_elements, out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, node_attrs: Tensor) -> Tensor:
        # [P, B, C, ...] =>  [B, C, P, ...]
        ndim = x.ndim
        device = x.device
        REST: List[int] = torch.arange(3, ndim, device=device).tolist()
        PERMUTE = [1, 2, 0] + REST
        x = x.permute(PERMUTE)

        idx = node_attrs.argmax(dim=-1)  # [B]
        W = self.weights[idx]  # [B, C, P]
        W = expand_dims_to(W, n_dim=ndim, dim=-1)
        out = (x * W).sum(dim=2) * self.alpha  # [B, C, ...]

        if self.bias is not None:
            b = self.bias[idx]  # [B, C]
            b = expand_dims_to(b, n_dim=out.ndim, dim=-1)
            out = out + b.view(-1, self.channel)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.num_path}, out_dim={self.channel}, bias={self.bias is not None})"
