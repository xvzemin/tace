################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch

from eqx import co3


class Linear(co3.Linear):
    """Equivariant linear map using flattened ``ir_mul`` features."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(irreps_in, irreps_out, biases=bias, **kwargs)


class ElementLinear(torch.nn.Module):
    """Element-conditioned equivariant linear map."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        num_elements: int,
        bias: bool = False,
        num_experts: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = co3.Irreps(irreps_in)
        self.irreps_out = co3.Irreps(irreps_out)
        self.num_elements = num_elements
        self.num_experts = num_experts or 1
        self.linear = co3.Linear(
            self.irreps_in,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
            biases=bias,
        )
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, self.linear.weight_numel)
        )
        if self.linear.bias_numel:
            self.bias = torch.nn.Parameter(
                torch.zeros(num_elements, self.linear.bias_numel)
            )
        else:
            self.register_buffer("bias", torch.empty(num_elements, 0))

    def forward(
        self,
        features: torch.Tensor,
        attrs: torch.Tensor,
    ) -> torch.Tensor:
        if attrs.ndim != 2 or attrs.size(-1) != self.num_elements:
            raise ValueError(
                f"ElementLinear attrs must have shape (batch, {self.num_elements})."
            )
        weight = torch.matmul(attrs.to(self.weight.dtype), self.weight)
        bias = torch.matmul(attrs.to(self.bias.dtype), self.bias)
        return self.linear(features, weight, bias)


class MoEElementLinear(torch.nn.Module):
    """Apply independent element-conditioned maps to channel experts."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        num_elements: int,
        num_experts: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_in = co3.Irreps(irreps_in)
        self.irreps_out = co3.Irreps(irreps_out)
        self.num_elements = num_elements
        self.num_experts = num_experts
        if num_experts < 1:
            raise ValueError("num_experts must be positive.")
        if any(mul % num_experts for _, mul in self.irreps_in + self.irreps_out):
            raise ValueError("Every multiplicity must be divisible by num_experts.")
        self.expert_irreps_in = co3.Irreps(
            [(ir, mul // num_experts) for ir, mul in self.irreps_in]
        )
        self.expert_irreps_out = co3.Irreps(
            [(ir, mul // num_experts) for ir, mul in self.irreps_out]
        )
        self.linear = co3.Linear(
            self.expert_irreps_in,
            self.expert_irreps_out,
            internal_weights=False,
            shared_weights=False,
            biases=bias,
        )
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, num_experts, self.linear.weight_numel)
        )
        if self.linear.bias_numel:
            self.bias = torch.nn.Parameter(
                torch.zeros(num_elements, num_experts, self.linear.bias_numel)
            )
        else:
            self.register_buffer("bias", torch.empty(num_elements, num_experts, 0))

    def _split_experts(self, features: torch.Tensor) -> torch.Tensor:
        entries = []
        for (ir, mul), ir_slice in zip(self.irreps_in, self.irreps_in.slices()):
            expert_mul = mul // self.num_experts
            entry = features[..., ir_slice].reshape(
                features.size(0), ir.dim, self.num_experts, expert_mul
            )
            entries.append(
                entry.permute(0, 2, 1, 3).reshape(
                    features.size(0), self.num_experts, ir.dim * expert_mul
                )
            )
        return torch.cat(entries, dim=-1)

    def _merge_experts(self, features: torch.Tensor) -> torch.Tensor:
        entries = []
        for (ir, mul), ir_slice in zip(
            self.expert_irreps_out, self.expert_irreps_out.slices()
        ):
            entry = features[..., ir_slice].reshape(
                features.size(0), self.num_experts, ir.dim, mul
            )
            entries.append(
                entry.permute(0, 2, 1, 3).reshape(
                    features.size(0), ir.dim * mul * self.num_experts
                )
            )
        return torch.cat(entries, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        attrs: torch.Tensor,
    ) -> torch.Tensor:
        if attrs.ndim != 2 or attrs.size(-1) != self.num_elements:
            raise ValueError(
                f"MoEElementLinear attrs must have shape (batch, {self.num_elements})."
            )
        node_type = attrs.argmax(dim=-1)
        weight = self.weight[node_type]
        bias = self.bias[node_type]
        output = self.linear(self._split_experts(features), weight, bias)
        return self._merge_experts(output)


__all__ = ["ElementLinear", "Linear", "MoEElementLinear"]
