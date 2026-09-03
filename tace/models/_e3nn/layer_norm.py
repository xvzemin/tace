################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, refactor this layernorm

from functools import partial

import torch
from e3nn import o3

_LAYER_NORM = [
    "merge_layer_norm",
    "merge_rms_norm",
]


def get_normalization_layer(
    norm_type,
    ls,
    num_channels,
    eps=1e-5,
    affine=True,
    normalization="component",
    irreps=None,
):
    assert norm_type in _LAYER_NORM
    if norm_type == "merge_layer_norm":
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == "merge_rms_norm":
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    else:
        raise ValueError
    return norm_class(
        ls,
        num_channels,
        eps,
        affine,
        normalization,
        irreps=irreps,
    )


class EquivariantMergeLayerNorm(torch.nn.Module):
    def __init__(
        self,
        ls,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
        centering=True,
        irreps=None,
    ):
        super().__init__()

        if irreps is not None:
            irreps = o3.Irreps(irreps)
            ls = [ir.l for _, ir in irreps]
        if isinstance(ls, int):
            self.ls = list(range(ls + 1))
        else:
            assert 0 in ls
            assert list(ls) == sorted(ls)
            self.ls = list(ls)

        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.centering = centering

        if irreps is None:
            self.scalar_index = 0
        else:
            offset = 0
            self.scalar_index = None
            for _, ir in irreps:
                if ir.is_scalar():
                    self.scalar_index = offset
                    break
                offset += ir.dim

        # total irreps dimension
        self.dim = sum([2 * l + 1 for l in self.ls])

        if self.affine:
            self.affine_weight = torch.nn.Parameter(
                torch.ones((len(self.ls), self.num_channels))
            )
            # build expand_index
            expand_index = torch.zeros(self.dim, dtype=torch.long)
            offset = 0
            for i, l in enumerate(self.ls):
                length = 2 * l + 1
                expand_index[offset : offset + length] = i
                offset += length
            self.register_buffer("expand_index", expand_index)
            if self.centering and self.scalar_index is not None:
                self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros(self.dim, 1)
            offset = 0
            for l in self.ls:
                length = 2 * l + 1
                balance_degree_weight[offset : offset + length, :] = 1.0 / length
                offset += length
            balance_degree_weight = balance_degree_weight / len(self.ls)
            balance_degree_weight = balance_degree_weight.permute(1, 0)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(ls={self.ls}, num_channels={self.num_channels}, "
            f"eps={self.eps}, std_balance_degrees={self.std_balance_degrees}, "
            f"centering={self.centering})"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # for L = 0
        if self.centering and self.scalar_index is not None:
            scalars = inputs.narrow(1, self.scalar_index, 1)
            scalars_mean = scalars.mean(dim=2, keepdim=True)  # [N, 1, 1]
            scalars = scalars - scalars_mean
            inputs = inputs.clone()
            inputs[:, self.scalar_index : self.scalar_index + 1, :] = scalars

        # for L >= 0
        feature_norm = inputs.pow(2)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
        if self.normalization == "norm":
            feature_norm = feature_norm.sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = torch.einsum(
                    "ai, nic -> nac",
                    self.balance_degree_weight,
                    feature_norm,
                )
            else:
                feature_norm = feature_norm.mean(dim=1, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, len(self.ls), self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        outputs = inputs * feature_norm

        if self.affine and self.centering and self.scalar_index is not None:
            scalar = outputs.narrow(1, self.scalar_index, 1)
            outputs = outputs.clone()
            outputs[:, self.scalar_index : self.scalar_index + 1, :] = (
                scalar + self.affine_bias.view(1, 1, self.num_channels)
            )

        return outputs
