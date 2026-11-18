"""Native fused symmetric-traceless products and differentiable adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.autograd import forward_ad_active
from .... import config
from ....common.cuda.cartesian import (
    cartesian_product,
    contract,
    pack_product,
)
from ....common.cuda.sparse_bilinear import evaluate as sparse_product
from ....common.irreducible_product import operands
from ....common.torch.sparse_bilinear import evaluate as inline_product
from ....torch.co3.ictp import tensor_product as torch_kernel
from . import permutation


def angular(module, x, y):
    if not x.is_cuda:
        return torch_kernel.angular(module, x, y)
    captured = torch.compiler.is_compiling() or get_proxy_mode() is not None
    if module.path_type == "symmetrize":
        if forward_ad_active() or (captured and config.ICTP_INLINE):
            return torch_kernel.axial_product(module, x, y)
        if module.l3 <= 3:
            return permutation.contract(x, y, [module.l1, module.l2, module.l3], 2)
        return permutation.orbit_product(
            x,
            y,
            [module.l1, module.l2, module.l3],
            module.indices,
            module.permutation,
            module.offsets,
            module.counts,
        )
    if module.path_type == "direct":
        if min(module.l1, module.l2) == 0:
            return x * y
        if module.l3 == 1 and module.l1 == module.l2:
            if forward_ad_active() or (captured and config.ICTP_INLINE):
                return torch_kernel.axial_product(module, x, y)
            return contract(x, y, [module.l1, module.l2, 1], 2)
        return cartesian_product(x, y, module.l1, module.l2, module.l3)
    if getattr(module, "fused_traces", False) and not captured:
        value = sparse_product(x, y, operands(module))
        return value.index_select(-1, module.bases[0].indices)
    if not getattr(module, "sparse_product", False):
        return torch_kernel.structured(
            module,
            x,
            y,
            cartesian_product,
            pack_product=None
            if forward_ad_active() or (captured and config.CARTESIAN_REDUCTION_INLINE)
            else pack_product,
        )
    if config.ICTP_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return inline_product(x, y, operands(module)).index_select(-1, module.indices)
    result = sparse_product(x, y, operands(module))
    return result.index_select(-1, module.indices)


def evaluate(module, x, y, weight):
    if module.packed_metadata is not None and not (
        config.ICTP_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        from ....common.channel_product import tensor_product

        result = tensor_product(
            module.packed_metadata, x, y, weight, module.packed_dim, backend="cuda"
        )
        if module.expansion_index is not None:
            result = result.index_select(-1, module.expansion_index)
        return result
    return torch_kernel.evaluate(module, x, y, weight)
