################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from eqx import co3

from .linear import ElementLinear, Linear


def get_resnet_layer(
    irreps_in,
    irreps_out,
    bias,
    num_elements,
    resnet_type,
):
    if resnet_type == "agnostic":
        return Linear(irreps_in, irreps_out, bias=bias)
    if resnet_type == "identity":
        return SkipIdentity(irreps_in, irreps_out)
    return ElementLinear(
        irreps_in,
        irreps_out,
        bias=bias,
        num_elements=num_elements,
    )


class SkipIdentity(torch.nn.Module):
    """Copy matching Cartesian entries and zero-pad all other outputs."""

    def __init__(self, irreps_in, irreps_out) -> None:
        super().__init__()
        self.irreps_in = co3.Irreps(irreps_in)
        self.irreps_out = co3.Irreps(irreps_out)
        self.instructions = []
        used = set()
        for i_in, (ir_in, mul_in) in enumerate(self.irreps_in):
            for i_out, (ir_out, mul_out) in enumerate(self.irreps_out):
                if i_out not in used and ir_in == ir_out and mul_in == mul_out:
                    self.instructions.append((i_in, i_out))
                    used.add(i_out)
                    break
        self.input_slices = self.irreps_in.slices()
        self.output_slices = self.irreps_out.slices()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = features.new_zeros(*features.shape[:-1], self.irreps_out.dim)
        for i_in, i_out in self.instructions:
            output[..., self.output_slices[i_out]] = features[
                ..., self.input_slices[i_in]
            ]
        return output


__all__ = ["SkipIdentity", "get_resnet_layer"]
