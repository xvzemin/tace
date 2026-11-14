"""Shared channel GEMMs followed by a native fixed angular readout."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda import sparse_bilinear, sparse_chain
from ...torch.o2 import o3_tensor_product as torch_kernel


def local(module, features, weight, harmonic_scale=None):
    if (
        not hasattr(module, "local_paths")
        or forward_ad_active()
        or (
            config.ALIGNED_PRODUCT_INLINE
            and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
        )
    ):
        return torch_kernel.local(module, features, weight, harmonic_scale)
    if getattr(module, "channelwise", False):
        return torch_kernel.direct(
            module, features, weight, harmonic_scale, sparse_bilinear.evaluate
        )
    return torch_kernel.program(
        module, features, weight, harmonic_scale, sparse_chain.evaluate
    )


def forward(module, features, wigner, wigner_inv, weight=None, harmonic_scale=None):
    return torch_kernel.forward(
        module, features, wigner, wigner_inv, weight, harmonic_scale
    )
