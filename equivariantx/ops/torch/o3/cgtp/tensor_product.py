"""Torch angular products, grouped channel GEMMs, and reference paths."""

import math

import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from .... import config


def evaluate(module, x, y, weight, supplied_weight):
    """Select the ordinary reference or optimized channel schedule."""
    if (
        not config.TORCH_REFERENCE
        and config.TORCH_GROUPED_CG
        and x.is_cuda
        and (0 < module.maximum_channels <= 16)
        and statically_known_true(math.prod(x.shape[:-1]) >= 512)
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return module.contraction(x, y, weight)
    return o3.TensorProduct.forward(module, x, y, weight if supplied_weight else None)
