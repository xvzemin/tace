###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, Union

import torch
from e3nn import o3

from eqx.conv import ACE
from tace.utils.env import acceleration_enabled

from ..linear import e3nnElementLinear, e3nnLinear, e3nnMoEElementLinear, has_lora
from ..mlp import ACTIVATION
from .base import Product
from .dropout import GraphDropPath
from .fused import uuuTensorProduct
from .paths import SymmetricProductPaths


class CgtpACE(Product):
    """Channel-wise ACE with element-dependent or element-independent coefficients."""

    def _setup(self):
        self.use_eqx = bool(acceleration_enabled("eqx"))
        for_coefs = {
            "irreps_out": self.irreps_coefs_out,
            "bias": self.use_bias,
        }
        coefs_cls = e3nnLinear if self.agnostic else e3nnElementLinear
        if not self.agnostic:
            for_coefs["num_elements"] = self.num_elements

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps(
                    [(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]
                ).simplify(),
                **for_coefs,
            )
        )
        self.irreps_base = self.irreps_hidden
        product_in1 = self.irreps_hidden
        symmetric_paths = (
            SymmetricProductPaths(product_in1) if self.correlation > 2 else None
        )
        for nu in range(2, self.correlation + 1):
            ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_base,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=self.l1l2,
                trainable=False,
                identical_inputs=nu == 2,
                warning=self.correlation > 2 and self.layer == 0 and not self.use_eqx,
                use_fused=self.correlation > 2
                and not self.use_time_reversal
                and not self.use_eqx,
                symmetric_paths=symmetric_paths,
                use_eqt=False if self.use_eqx else None,
            )
            self.aces.append(ace)
            self.coefs.append(
                coefs_cls(
                    o3.Irreps(
                        [(self.num_hidden_channel, ir) for _, ir in ace.irreps_out]
                    ).simplify(),
                    **for_coefs,
                )
            )
            product_in1 = ace.irreps_out

        if self.use_eqx and self.aces:
            self.eqx_ace = ACE(
                [ace.tp for ace in self.aces],
                [coef.linear for coef in self.coefs],
            )

        self.linear_up = (
            e3nnLinear(
                self.irreps_in, self.irreps_hidden, bias=self.use_bias
            )
            if self.num_channel != self.num_hidden_channel
            else torch.nn.Identity()
        )
        self.linear = e3nnLinear(
            o3.Irreps(
                [(self.num_hidden_channel, ir) for _, ir in self.irreps_coefs_out]
            ),
            self.irreps_out,
            bias=self.use_bias,
        )
        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        sc: torch.Tensor,
        batch: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.agnostic:
            for_coefs = {}
        else:
            if node_type is None:
                node_type = node_attrs.argmax(dim=-1)
            for_coefs = {"attrs": node_attrs, "node_type": node_type}

        node_feats = self.linear_up(node_feats)
        corr_feats = node_feats
        if (
            hasattr(self, "eqx_ace")
            and node_feats.is_cuda
            and not any(has_lora(coef) for coef in self.coefs)
        ):
            weights = []
            for coef in self.coefs:
                if coef.use_matrix_weight:
                    rows = 1 if self.agnostic else self.num_elements
                    weight = torch.cat(
                        [w.reshape(rows, -1) for w in coef.weight], dim=-1
                    )
                else:
                    weight = coef.weight.reshape(
                        1 if self.agnostic else self.num_elements, -1
                    )
                weights.append(weight)
            types = (
                torch.zeros(
                    node_feats.shape[0], dtype=torch.long, device=node_feats.device
                )
                if self.agnostic
                else node_type
            )
            outs = self.eqx_ace(node_feats, weights, types)
            for coef in self.coefs:
                if coef.bias is not None:
                    bias = (
                        coef.bias
                        if self.agnostic
                        else node_attrs @ coef.bias.reshape(self.num_elements, -1)
                    )
                    outs = (
                        outs + bias.index_select(-1, coef._bias_index) * coef._bias_mask
                    )
        else:
            outs = self.coefs[0](corr_feats, **for_coefs)
            for index, ace in enumerate(self.aces):
                corr_feats = ace(corr_feats, node_feats)
                outs = outs + self.coefs[index + 1](corr_feats, **for_coefs)

        outs = self.linear(outs)
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        if sc is not None:
            outs = outs + sc
        return outs


class BilinearMoEACE(Product):
    """ACE with optional bilinear gates, coefficient experts and a shared expert."""

    def _setup(self):

        self.scale = 1.0 / math.sqrt(2.0)

        for_coefs = {
            "irreps_out": self.irreps_coefs_out,
            "bias": self.use_bias,
        }
        if self.agnostic:
            coefs_cls = e3nnLinear
        else:
            for_coefs["num_elements"] = self.num_elements
            coefs_cls = e3nnElementLinear
            if self.num_expert > 1:
                coefs_cls = e3nnMoEElementLinear
                for_coefs["num_experts"] = self.num_expert

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        if self.use_shared_expert and self.num_expert > 1:
            self.shared_coefs = torch.nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps(
                    [(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]
                ).simplify(),
                **for_coefs,
            )
        )
        if hasattr(self, "shared_coefs"):
            self.shared_coefs.append(
                e3nnLinear(
                    o3.Irreps(
                        [(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]
                    ).simplify(),
                    self.irreps_coefs_out,
                    bias=self.use_bias,
                )
            )

        self.irreps_base = self.irreps_hidden
        if self.use_bilinear_gate:
            assert self.correlation == 2
            self.nonlinearity = ACTIVATION[self.scalar_act]()
            self.irreps_base = self.irreps_hidden + o3.Irreps(
                [(self.num_hidden_channel, o3.Irrep("0e"))]
            )

        product_in1 = self.irreps_hidden
        symmetric_paths = (
            SymmetricProductPaths(product_in1)
            if self.correlation > 2 and not self.use_bilinear_gate
            else None
        )

        for nu in range(2, self.correlation + 1):
            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_base,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=self.l1l2,
                trainable=self.use_bilinear_gate,
                identical_inputs=nu == 2 and not self.use_bilinear_gate,
                warning=self.correlation > 2 and self.layer == 0,
                use_fused=self.correlation > 2 and not self.use_time_reversal,
                symmetric_paths=symmetric_paths,
            )
            self.aces.append(this_ace)
            self.coefs.append(
                coefs_cls(
                    o3.Irreps(
                        [(self.num_hidden_channel, ir) for _, ir in this_ace.irreps_out]
                    ).simplify(),
                    **for_coefs,
                )
            )
            if hasattr(self, "shared_coefs"):
                self.shared_coefs.append(
                    e3nnLinear(
                        o3.Irreps(
                            [
                                (self.num_hidden_channel, ir)
                                for _, ir in this_ace.irreps_out
                            ]
                        ).simplify(),
                        self.irreps_coefs_out,
                        bias=self.use_bias,
                    )
                )
            product_in1 = this_ace.irreps_out

        if self.use_bilinear_gate:
            self._ace_gate_slices = []
            gate_offset = 0
            for ace in self.aces:
                self._ace_gate_slices.append(
                    slice(gate_offset, gate_offset + ace.weight_numel)
                )
                gate_offset += ace.weight_numel
            self.num_ace_gate_weights = gate_offset

            self.irreps_double_hidden = o3.Irreps(
                [(2 * mul, ir) for mul, ir in self.irreps_hidden]
            )
            self.irreps_linear_up = (
                o3.Irreps(f"{self.num_ace_gate_weights}x0e") + self.irreps_double_hidden
            )
            self.linear_up = e3nnLinear(
                self.irreps_in,
                self.irreps_linear_up,
                bias=self.use_bias,
            )
            self._double_hidden_slices = list(self.irreps_double_hidden.slices())
            self._hidden_muls = [mul for mul, _ in self.irreps_hidden]
            self._hidden_ir_dims = [ir.dim for _, ir in self.irreps_hidden]
        else:
            self.linear_up = (
                e3nnLinear(
                    self.irreps_in,
                    self.irreps_hidden,
                    bias=self.use_bias,
                )
                if self.num_channel != self.num_hidden_channel
                else torch.nn.Identity()
            )

        self.linear = e3nnLinear(
            o3.Irreps(
                [(self.num_hidden_channel, ir) for _, ir in self.irreps_coefs_out]
            ),
            self.irreps_out,
            bias=self.use_bias,
        )

        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

    def _merge_shared_expert(
        self,
        grouped: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        return (grouped + shared) * self.scale

    def _linear_up_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        x = self.linear_up(x)
        if not self.use_bilinear_gate:
            return x, None, None

        ace_weights = self.nonlinearity(x[:, : self.num_ace_gate_weights])
        double_features = x[:, self.num_ace_gate_weights :]

        node_fields = []
        base_fields = []
        for tensor_slice, mul, ir_dim in zip(
            self._double_hidden_slices,
            self._hidden_muls,
            self._hidden_ir_dims,
        ):
            field = double_features[:, tensor_slice].reshape(
                x.shape[0],
                2 * mul,
                ir_dim,
            )
            node_fields.append(field[:, :mul].reshape(x.shape[0], -1))
            base_fields.append(field[:, mul:].reshape(x.shape[0], -1))

        node_feats = torch.cat(node_fields, dim=-1)
        base_feats = torch.cat(base_fields, dim=-1)
        base_feats = torch.cat(
            (
                base_feats,
                base_feats.new_ones(base_feats.shape[0], self.num_hidden_channel),
            ),
            dim=-1,
        )
        return node_feats, base_feats, ace_weights

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        sc: torch.Tensor,
        batch: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.agnostic:
            for_coefs = {}
        else:
            if node_type is None:
                node_type = node_attrs.argmax(dim=-1)
            for_coefs = {"attrs": node_attrs, "node_type": node_type}

        node_feats, base_feats, ace_weights = self._linear_up_features(node_feats)

        corr_feats = {
            1: node_feats,
        }

        outs = self.coefs[0](corr_feats[1], **for_coefs)
        shared_outs = (
            self.shared_coefs[0](corr_feats[1])
            if hasattr(self, "shared_coefs")
            else None
        )

        for nu in range(2, self.correlation + 1):
            if self.use_bilinear_gate:
                corr_feats[nu] = self.aces[nu - 2](
                    corr_feats[nu - 1],
                    base_feats,
                    ace_weights[:, self._ace_gate_slices[nu - 2]],
                )
            else:
                corr_feats[nu] = self.aces[nu - 2](corr_feats[nu - 1], node_feats)
            outs = outs + self.coefs[nu - 1](corr_feats[nu], **for_coefs)
            if shared_outs is not None:
                shared_outs = shared_outs + self.shared_coefs[nu - 1](corr_feats[nu])

        if shared_outs is not None:
            outs = self._merge_shared_expert(outs, shared_outs)

        outs = self.linear(outs)

        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)

        if sc is not None:
            outs = outs + sc

        return outs


PRODUCT: Dict[str, torch.nn.Module] = {
    "cgtp": CgtpACE,
}

product_aliases = {
    "coupled": "cgtp",
}

PRODUCT.update(
    {
        alias: PRODUCT[target]
        for alias, target in product_aliases.items()
    }
)
