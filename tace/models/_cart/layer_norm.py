################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from eqx import co3


def get_normalization_layer(
    norm_type,
    irreps,
    eps=1.0e-5,
    affine=True,
    normalization="component",
):
    if norm_type not in ("merge_layer_norm", "merge_rms_norm"):
        raise ValueError(f"Unknown Cartesian normalization: {norm_type!r}.")
    return EquivariantMergeLayerNorm(
        irreps,
        eps=eps,
        affine=affine,
        normalization=normalization,
        centering=norm_type == "merge_layer_norm",
    )


class EquivariantMergeLayerNorm(torch.nn.Module):
    """Normalize equal-multiplicity Cartesian entries channel-wise."""

    def __init__(
        self,
        irreps,
        eps=1.0e-5,
        affine=True,
        normalization="component",
        centering=True,
    ) -> None:
        super().__init__()
        self.irreps = co3.Irreps(irreps)
        multiplicities = {mul for _, mul in self.irreps}
        if len(multiplicities) != 1:
            raise ValueError(
                "Cartesian merge normalization requires equal multiplicities."
            )
        self.num_channels = next(iter(multiplicities))
        self.eps = eps
        self.affine = affine
        self.normalization = normalization
        self.centering = centering
        self.slices = self.irreps.slices()
        if affine:
            self.weight = torch.nn.Parameter(
                torch.ones(len(self.irreps), self.num_channels)
            )
            self.bias = torch.nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        entries = []
        moments = []
        for (ir, mul), ir_slice in zip(self.irreps, self.slices):
            values = features[..., ir_slice].reshape(*features.shape[:-1], ir.dim, mul)
            if self.centering and ir.is_even_scalar():
                values = values - values.mean(dim=-1, keepdim=True)
            entries.append(values)
            divisor = ir.dof if self.normalization == "component" else 1
            moments.append(values.square().sum(dim=-2) / divisor)
        scale = torch.stack(moments, dim=-2).mean(dim=-2, keepdim=True)
        scale = (scale + self.eps).rsqrt()
        outputs = []
        for index, ((ir, mul), values) in enumerate(zip(self.irreps, entries)):
            output = values * scale
            if self.weight is not None:
                output = output * self.weight[index]
                if self.centering and ir.is_even_scalar():
                    output = output + self.bias
            outputs.append(output.reshape(*features.shape[:-1], ir.dim * mul))
        return torch.cat(outputs, dim=-1)


__all__ = ["EquivariantMergeLayerNorm", "get_normalization_layer"]
