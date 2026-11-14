"""Native O(2) feature packing with a shared Fourier GEMM."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.grid import evaluate
from ...torch.o2 import o2grid as torch_kernel


def inline():
    return forward_ad_active() or (
        config.O2_GRID_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


def to_grid(module, features, features2=None):
    if inline():
        return torch_kernel.to_grid(module, features, features2)
    return torch_kernel.to_grid_program(module, features, features2, evaluate)


def from_grid(module, features, split=False):
    if inline():
        return torch_kernel.from_grid(module, features, split)
    return torch_kernel.from_grid_program(module, features, split, evaluate)
