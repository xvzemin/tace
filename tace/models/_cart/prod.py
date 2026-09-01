################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Optional

import torch

from eqx import co3

from ..mlp import get_scaled_activation
from .base import Product, possible_irreps
from .dropout import GraphDropPath
from .fused import uuuTensorProduct
from .linear import ElementLinear, Linear, MoEElementLinear


class CgtpACE(Product):
    """Cartesian many-body expansion based on repeated channel-wise products."""

    def _make_coefficient(self, irreps_in):
        if self.agnostic:
            return Linear(irreps_in, self.irreps_coefficients, bias=self.use_bias)
        if self.num_expert > 1:
            return MoEElementLinear(
                irreps_in,
                self.irreps_coefficients,
                bias=self.use_bias,
                num_elements=self.num_elements,
                num_experts=self.num_expert,
            )
        return ElementLinear(
            irreps_in,
            self.irreps_coefficients,
            bias=self.use_bias,
            num_elements=self.num_elements,
            num_experts=self.num_expert,
        )

    def _setup(self) -> None:
        if self.nonlinear not in (None, "bilinear_gate"):
            raise ValueError(
                "Cartesian product nonlinear must be None or 'bilinear_gate'."
            )
        self.use_bilinear_gate = self.nonlinear == "bilinear_gate"
        if self.use_bilinear_gate and self.correlation != 2:
            raise ValueError("bilinear_gate requires correlation=2.")
        self.irreps_coefficients = co3.Irreps(
            [(ir, self.num_hidden_channel) for ir, _ in self.irreps_out]
        )
        hidden_types = co3.Irreps([(ir, 1) for ir, _ in self.irreps_in]).regroup()
        self.irreps_hidden = co3.Irreps(
            [(ir, self.num_hidden_channel) for ir, _ in hidden_types]
        )
        self.aces = torch.nn.ModuleList()
        correlation_irreps = [self.irreps_hidden]
        product_in = self.irreps_hidden
        base_irreps = (
            self.irreps_hidden
            + co3.Irreps([(co3.Irrep("0e"), self.num_hidden_channel)])
            if self.use_bilinear_gate
            else self.irreps_hidden
        )
        for nu in range(2, self.correlation + 1):
            if nu == self.correlation and self.layer == self.num_layers - 1:
                output_types = self.target_irreps
            else:
                max_degree = self.Lmax if nu == self.correlation else self.lmax
                output_types = possible_irreps(
                    product_in,
                    self.irreps_hidden,
                    parity=self.parity,
                    lmax=max_degree,
                )
            irreps_out = co3.Irreps(
                [(ir, self.num_hidden_channel) for ir, _ in output_types]
            )
            ace = uuuTensorProduct(
                product_in,
                base_irreps,
                irreps_out,
                l1l2=self.l1l2,
                trainable=self.use_bilinear_gate,
            )
            self.aces.append(ace)
            correlation_irreps.append(irreps_out)
            product_in = irreps_out
        if self.use_bilinear_gate:
            self.gate_slices = []
            offset = 0
            for ace in self.aces:
                self.gate_slices.append(slice(offset, offset + ace.weight_numel))
                offset += ace.weight_numel
            self.num_gate_weights = offset
            doubled = co3.Irreps([(ir, 2 * mul) for ir, mul in self.irreps_hidden])
            self.linear_up = Linear(
                self.irreps_in,
                co3.Irreps([(co3.Irrep("0e"), offset)]) + doubled,
                bias=self.use_bias,
            )
            self.gate_activation = get_scaled_activation(self.scalar_act)
        else:
            self.linear_up = Linear(
                self.irreps_in,
                self.irreps_hidden,
                bias=self.use_bias,
            )
        self.coefficients = torch.nn.ModuleList(
            self._make_coefficient(irreps) for irreps in correlation_irreps
        )
        if self.use_shared_expert and self.num_expert > 1:
            self.shared_coefficients = torch.nn.ModuleList(
                Linear(irreps, self.irreps_coefficients, bias=self.use_bias)
                for irreps in correlation_irreps
            )
        self.linear_down = Linear(
            self.irreps_coefficients,
            self.irreps_out,
            bias=self.use_bias,
        )
        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def _coefficient(
        self,
        module: torch.nn.Module,
        features: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        if self.agnostic:
            return module(features)
        return module(features, node_attrs)

    def _linear_up(self, features: torch.Tensor):
        features = self.linear_up(features)
        if not self.use_bilinear_gate:
            return features, features, None
        weights = self.gate_activation(features[..., : self.num_gate_weights])
        doubled = features[..., self.num_gate_weights :]
        node_entries = []
        base_entries = []
        for (ir, mul), ir_slice in zip(
            self.irreps_hidden,
            co3.Irreps([(ir, 2 * mul) for ir, mul in self.irreps_hidden]).slices(),
        ):
            entry = doubled[..., ir_slice].reshape(doubled.size(0), ir.dim, 2 * mul)
            node_entries.append(entry[..., :mul].reshape(doubled.size(0), -1))
            base_entries.append(entry[..., mul:].reshape(doubled.size(0), -1))
        node_features = torch.cat(node_entries, dim=-1)
        base_features = torch.cat(base_entries, dim=-1)
        base_features = torch.cat(
            (
                base_features,
                base_features.new_ones(base_features.size(0), self.num_hidden_channel),
            ),
            dim=-1,
        )
        return node_features, base_features, weights

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        sc: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        node_feats, base, gate_weights = self._linear_up(node_feats)
        correlations = [node_feats]
        for index, ace in enumerate(self.aces):
            weights = (
                gate_weights[..., self.gate_slices[index]]
                if gate_weights is not None
                else None
            )
            correlations.append(ace(correlations[-1], base, weights))
        output = sum(
            (
                self._coefficient(module, features, node_attrs)
                for module, features in zip(self.coefficients, correlations)
            )
        )
        if hasattr(self, "shared_coefficients"):
            shared = sum(
                module(features)
                for module, features in zip(self.shared_coefficients, correlations)
            )
            output = (output + shared) / math.sqrt(2.0)
        output = self.linear_down(output)
        if hasattr(self, "stochastic_depth"):
            output = self.stochastic_depth(output, batch)
        if sc is not None:
            output = output + sc
        return output


PRODUCT = {
    "spatial": CgtpACE,
    "coupled": CgtpACE,
    "cgtp": CgtpACE,
    "glu": CgtpACE,
}


__all__ = ["CgtpACE", "PRODUCT"]
