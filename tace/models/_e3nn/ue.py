################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch
from e3nn import o3
from e3nn.nn import Activation

from ...dataset.quantity import PROPERTY
from ..linear import e3nnElementLinear, e3nnLinear
from ..mlp import ACTIVATION, MLP
from ..time_reversal import with_time_reversal


def _property_irreps(name: str, time_reversal: bool) -> o3.Irreps:
    irreps = o3.Irreps(PROPERTY[name]["irreps"])
    if not time_reversal:
        return irreps
    return with_time_reversal(
        irreps,
        PROPERTY[name].get("time_reversal", 1),
    )


class UniversalInvariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        invariant_embedding: dict[str, Union[bool, str, int]],
        bias: bool,
        activation: str,
    ):
        super().__init__()

        self.uie = torch.nn.ModuleDict()

        total_dim = 0
        for k, v in invariant_embedding.items():
            p_type = PROPERTY[k]["type"]
            if p_type == "int":
                self.uie[k] = torch.nn.Embedding(v["num_embeddings"], out_dim)
            elif p_type == "float":
                self.uie[k] = MLP([1, out_dim, out_dim], bias=bias)
            else:
                raise
            total_dim += out_dim

        self.project = MLP(
            [total_dim, out_dim],
            bias=bias,
            layer_norm=False,
        )
        self.act = Activation(f"{out_dim}x0e", [ACTIVATION[activation]()])

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:

        batch = data["batch"]
        embeddings = []

        for p, module in self.uie.items():
            p_type = PROPERTY[p]["type"]
            p_scope = PROPERTY[p]["scope"]

            attr = data[p]
            if p_scope == "per-system":
                attr = attr[batch]
            if p_type == "float":
                attr = attr.unsqueeze(-1)

            embeddings.append(module(attr))

        return self.act(self.project(torch.cat(embeddings, dim=-1))).unsqueeze(1)


class UniversalEquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_channel: int,
        num_elements: int,
        config: dict[str, Union[bool, str, int]],
        time_reversal: bool = True,
    ):
        super().__init__()

        self.config = config
        self.irreps_in = irreps_in
        irreps_out = irreps_in
        for p in config.keys():
            irreps_out += _property_irreps(p, time_reversal)
        irreps_out = irreps_out.regroup()
        self.irreps_out = o3.Irreps([(num_channel, ir) for _, ir in irreps_out])

        self.linear = e3nnLinear(self.irreps_in, self.irreps_out)
        self.uee = torch.nn.ModuleDict()
        for k, v in config.items():
            self.uee[k] = e3nnElementLinear(
                _property_irreps(k, time_reversal),
                self.irreps_out,
                bias=True,
                num_elements=num_elements,
            )
            self.config[k]["scope"] = PROPERTY[k]["scope"]

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        data: dict[str, torch.Tensor],
    ) -> torch.Tensor:

        node_feats = self.linear(node_feats)

        batch = data["batch"]
        for p, e_linear in self.uee.items():
            scope = self.config[p]["scope"]
            normalizer = self.config[p]["normalizer"]
            attr = data[p] * normalizer
            if scope == "per-system":
                attr = attr[batch]
            if PROPERTY[p]["rank"] == 0:
                attr = attr.unsqueeze(-1)
            uee_feats = e_linear(attr, node_attrs)
            node_feats = node_feats + uee_feats

        return node_feats

    def extra_repr(self):
        return str(self.config)
