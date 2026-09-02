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


class IctpACE(Product):
    """Many-body expansion based on repeated channel-wise products."""

    def _make_coefficient(self, irreps_in):
        if self.agnostic:
            return Linear(irreps_in, self.irreps_coefs_out, bias=self.use_bias)
        if self.num_expert > 1:
            return MoEElementLinear(
                irreps_in,
                self.irreps_coefs_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
                num_experts=self.num_expert,
            )
        return ElementLinear(
            irreps_in,
            self.irreps_coefs_out,
            bias=self.use_bias,
            num_elements=self.num_elements,
            num_experts=self.num_expert,
        )

    def _setup(self) -> None:
        if self.nonlinear not in (None, "bilinear_gate"):
            raise ValueError(
                "Product nonlinear must be None or 'bilinear_gate'."
            )
        self.use_bilinear_gate = self.nonlinear == "bilinear_gate"
        if self.use_bilinear_gate and self.correlation != 2:
            raise ValueError("bilinear_gate requires correlation=2.")
        self.scale = 1.0 / math.sqrt(2.0)
        self.irreps_coefs_out = co3.Irreps(
            [(ir, self.num_hidden_channel) for ir, _ in self.irreps_out]
        )
        hidden_types = co3.Irreps([(ir, 1) for ir, _ in self.irreps_in]).regroup()
        self.irreps_hidden = co3.Irreps(
            [(ir, self.num_hidden_channel) for ir, _ in hidden_types]
        )
        self.aces = torch.nn.ModuleList()
        correlation_irreps = [self.irreps_hidden]
        product_in = self.irreps_hidden
        self.irreps_base = (
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
                self.irreps_base,
                irreps_out,
                l1l2=self.l1l2,
                trainable=self.use_bilinear_gate,
                identical_inputs=nu == 2 and not self.use_bilinear_gate,
            )
            self.aces.append(ace)
            correlation_irreps.append(ace.irreps_out)
            product_in = ace.path_irreps_out
        if self.use_bilinear_gate:
            self._ace_gate_slices = []
            offset = 0
            for ace in self.aces:
                self._ace_gate_slices.append(
                    slice(offset, offset + ace.weight_numel)
                )
                offset += ace.weight_numel
            self.num_ace_gate_weights = offset
            self.irreps_double_hidden = co3.Irreps(
                [(ir, 2 * mul) for ir, mul in self.irreps_hidden]
            )
            self.irreps_linear_up = (
                co3.Irreps([(co3.Irrep("0e"), offset)])
                + self.irreps_double_hidden
            )
            self.linear_up = Linear(
                self.irreps_in,
                self.irreps_linear_up,
                bias=self.use_bias,
            )
            self.nonlinearity = get_scaled_activation(self.scalar_act)
        else:
            self.linear_up = (
                Linear(
                    self.irreps_in,
                    self.irreps_hidden,
                    bias=self.use_bias,
                )
                if self.num_channel != self.num_hidden_channel
                else torch.nn.Identity()
            )
        self.coefs = torch.nn.ModuleList(
            self._make_coefficient(irreps) for irreps in correlation_irreps
        )
        if self.use_shared_expert and self.num_expert > 1:
            self.shared_coefs = torch.nn.ModuleList(
                Linear(irreps, self.irreps_coefs_out, bias=self.use_bias)
                for irreps in correlation_irreps
            )
        self.linear = Linear(
            self.irreps_coefs_out,
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

    def _merge_shared_expert(
        self,
        grouped: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        return (grouped + shared) * self.scale

    def _linear_up_features(self, features: torch.Tensor):
        features = self.linear_up(features)
        if not self.use_bilinear_gate:
            return features, features, None
        weights = self.nonlinearity(features[..., : self.num_ace_gate_weights])
        doubled = features[..., self.num_ace_gate_weights :]
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
        node_feats, base_feats, ace_weights = self._linear_up_features(node_feats)
        corr_feats = {1: node_feats}
        for nu in range(2, self.correlation + 1):
            weights = None
            if ace_weights is not None:
                weights = ace_weights[..., self._ace_gate_slices[nu - 2]]
            corr_feats[nu] = self.aces[nu - 2](
                corr_feats[nu - 1], base_feats, weights
            )
        outs = sum(
            self._coefficient(module, features, node_attrs)
            for module, features in zip(self.coefs, corr_feats.values())
        )
        if hasattr(self, "shared_coefs"):
            shared_outs = sum(
                module(features)
                for module, features in zip(self.shared_coefs, corr_feats.values())
            )
            outs = self._merge_shared_expert(outs, shared_outs)
        outs = self.linear(outs)
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        if sc is not None:
            outs = outs + sc
        return outs


PRODUCT = {
    "ictp": IctpACE,
}


__all__ = ["IctpACE", "PRODUCT"]
