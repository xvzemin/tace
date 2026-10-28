################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

from ..linear import e3nnElementLinear, e3nnLinear


def get_resnet_layer(
    irreps_in,
    irreps_out,
    bias,
    num_elements,
    resnet_type,
):
    if resnet_type == "agnostic":
        return e3nnLinear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            bias=bias,
        )
    elif resnet_type == "identity":
        return SkipIdentity(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

    return e3nnElementLinear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        bias=bias,
        num_elements=num_elements,
    )


class ProjectUp(torch.nn.Module):
    """From https://github.com/SamsungDS/GGNN/blob/main/GGNN/model/EquFlashV2/nn/skip.py"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        idx = self._get_indices(self.irreps_in, self.irreps_out)
        self.register_buffer("idx", idx, persistent=False)

    @staticmethod
    def _get_indices(irreps_in, irreps_out):
        idxs = []
        out_slices = list(irreps_out.slices())
        used = set()

        for mul_ir in irreps_in:
            for index, (target, target_slice) in enumerate(zip(irreps_out, out_slices)):
                if index not in used and mul_ir == target:
                    idxs.append(torch.arange(target_slice.start, target_slice.stop))
                    used.add(index)
                    break
            else:
                raise ValueError(
                    f"{irreps_in} is not a sub-representation of {irreps_out}"
                )

        return torch.cat(idxs)

    def forward(self, x):
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        out[:, self.idx] = x
        return out


class ProjectDown(torch.nn.Module):
    """From https://github.com/SamsungDS/GGNN/blob/main/GGNN/model/EquFlashV2/nn/skip.py"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        idx = ProjectUp._get_indices(self.irreps_out, self.irreps_in)
        self.register_buffer("idx", idx, persistent=False)

    def forward(self, x):
        return x[:, self.idx]


class SkipIdentity(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        if self.irreps_in == self.irreps_out:
            self.proj = torch.nn.Identity()
        elif self.irreps_in.dim < self.irreps_out.dim:
            self.proj = ProjectUp(self.irreps_in, self.irreps_out)
        else:
            self.proj = ProjectDown(self.irreps_in, self.irreps_out)

    def forward(self, x):
        return self.proj(x)
