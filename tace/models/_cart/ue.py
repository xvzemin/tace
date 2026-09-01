################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch

from eqx import co3

from ...dataset.quantity import PROPERTY
from ..mlp import MLP, get_scaled_activation
from .linear import ElementLinear, Linear


class UniversalInvariantEmbedding(torch.nn.Module):
    def __init__(self, out_dim, invariant_embedding, bias, activation="silu"):
        super().__init__()
        self.out_dim = out_dim
        self.embeddings = torch.nn.ModuleDict()
        for name, config in invariant_embedding.items():
            if PROPERTY[name]["type"] == "int":
                self.embeddings[name] = torch.nn.Embedding(
                    config["num_embeddings"], out_dim
                )
            else:
                self.embeddings[name] = MLP([1, out_dim, out_dim], bias=bias)
        self.project = MLP(
            [max(1, len(self.embeddings)) * out_dim, out_dim],
            bias=bias,
        )
        self.activation = get_scaled_activation(activation)

    def forward(self, data):
        batch = data["batch"]
        outputs = []
        for name, module in self.embeddings.items():
            values = data[name]
            if PROPERTY[name]["scope"] == "per-system":
                values = values[batch]
            if PROPERTY[name]["type"] == "float":
                values = values.unsqueeze(-1)
            outputs.append(module(values))
        return self.activation(self.project(torch.cat(outputs, dim=-1)))


class UniversalEquivariantEmbedding(torch.nn.Module):
    def __init__(self, irreps_in, num_channel, num_elements, config):
        super().__init__()
        self.config = config
        self.irreps_in = co3.Irreps(irreps_in)
        types = self.irreps_in + sum(
            (co3.Irreps(PROPERTY[name]["irreps"]) for name in config),
            co3.Irreps(),
        )
        types = types.regroup()
        self.irreps_out = co3.Irreps([(ir, num_channel) for ir, _ in types])
        self.linear = Linear(self.irreps_in, self.irreps_out)
        self.embeddings = torch.nn.ModuleDict(
            {
                name: ElementLinear(
                    co3.Irreps(PROPERTY[name]["irreps"]),
                    self.irreps_out,
                    bias=True,
                    num_elements=num_elements,
                )
                for name in config
            }
        )

    def forward(self, node_feats, node_attrs, data):
        output = self.linear(node_feats)
        batch = data["batch"]
        for name, module in self.embeddings.items():
            values = data[name] * self.config[name].get("normalizer", 1.0)
            if PROPERTY[name]["scope"] == "per-system":
                values = values[batch]
            values = values.reshape(values.size(0), -1)
            irreps = co3.Irreps(PROPERTY[name]["irreps"])
            if values.size(-1) != irreps.dim:
                raise ValueError(
                    f"Universal Cartesian property {name!r} has trailing size "
                    f"{values.size(-1)}, expected {irreps.dim}."
                )
            projected = []
            for (ir, mul), ir_slice in zip(irreps, irreps.slices()):
                field = values[..., ir_slice].reshape(values.size(0), ir.dim, mul)
                projected.append(
                    co3.project(field, ir.l).reshape(values.size(0), ir.dim * mul)
                )
            output = output + module(torch.cat(projected, dim=-1), node_attrs)
        return output


__all__ = ["UniversalEquivariantEmbedding", "UniversalInvariantEmbedding"]
