"""Native staged Cartesian transforms with an optional inline Torch schedule."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ..torch import ictd as torch_kernel
from .grid import linear
from .sparse_chain import evaluate, packed


def inline():
    return forward_ad_active() or (
        config.ICTD_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


def forward(module, features):
    if inline():
        return torch_kernel.forward(module, features)
    return transform(module, features, False)


def inverse(module, features):
    if inline():
        return torch_kernel.inverse(module, features)
    return transform(module, features, True)


def transform(module, features, reverse):
    if (
        hasattr(module, "packed_outer")
        and module.dim <= config.ICTD_FUSED_MAX_WIDTH
        and features.is_contiguous()
        and not module.packed_weights.requires_grad
    ):
        return packed(
            features,
            module.packed_indices,
            module.packed_weights,
            list(module.packed_outer),
            list(module.packed_terms),
            list(module.packed_offsets),
            reverse,
        )
    if torch_kernel.use_dense(module, features):
        matrix = module.change_of_basis
        return linear(features, matrix if reverse else matrix.T)
    return torch_kernel.transform(module, features, reverse, evaluate)


def project(module, features, index):
    if inline():
        return torch_kernel.project(module, features, index)
    if torch_kernel.use_dense(module, features):
        return torch_kernel.dense_project(module, features, index, linear)
    return torch_kernel.projection(module, features, index, evaluate)
