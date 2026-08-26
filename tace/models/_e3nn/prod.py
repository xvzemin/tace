###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, Union

import torch
from e3nn import o3

from ..linear import e3nnElementLinear, e3nnLinear, e3nnMoEElementLinear
from ..mlp import ACTIVATION
from .base import Product
from .dropout import GraphDropPath
from .fused import uuuTensorProduct


class CgtpACE(Product):
    """
    The most expressive ACE implementation based on Clebsch-Gordan tensor products.

    This class computes all possible many-body tensor product paths and couples
    all channels, forming a highly expressive product basis.

    Note:
        It is recommended to use no more than 64 channels for each expert, as
        increasing the number of channels beyond this does not necessarily lead
        to better performance and may introduce unnecessary computational overhead.
    """

    def _setup(self):

        if self.parity and self.correlation > 2:
            raise ValueError(
                "CgtpACE with parity=True currently requires correlation < 3. "
                "Invalid paths for correlation >= 3 have not yet been filtered "
                "and may cause redundant computation, so higher-order full O(3) "
                "products are temporarily disabled."
            )

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

        for nu in range(2, self.correlation + 1):
            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_base,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=self.l1l2,
                trainable=self.use_bilinear_gate,
                identical_inputs=nu == 2 and not self.use_bilinear_gate,
                warning=self.correlation > 2 and self.layer == 0,
                use_fused=self.correlation > 2,
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
    ) -> torch.Tensor:

        if self.agnostic:
            for_coefs = {}
        else:
            for_coefs = {"attrs": node_attrs}

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
    "spatial": CgtpACE,
    "coupled": CgtpACE,
    "cgtp": CgtpACE,
    "glu": CgtpACE,
}
