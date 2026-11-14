"""Native spherical quadratures using the shared channel GEMM."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda import grid
from ...torch.o3 import s2grid as torch_kernel


def inline():
    return forward_ad_active() or (
        config.S2_GRID_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


def to_grid(module, features):
    if inline():
        return torch_kernel.to_grid(module, features)
    return grid.linear(features, module.synthesis)


def from_grid(module, features):
    if inline():
        return torch_kernel.from_grid(module, features)
    return grid.linear(features, module.analysis)
