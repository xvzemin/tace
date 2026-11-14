"""Native fused symmetric-traceless products and differentiable adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config, get_kernel
from ....common.cuda.sparse_bilinear import evaluate as sparse_product
from ....common.irreducible_product import operands
from ....torch.co3.ictp import tensor_product as torch_kernel


def angular(module, x, y):
    if not x.is_cuda:
        return torch_kernel.angular(module, x, y)
    if config.ICTP_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return torch_kernel.angular(module, x, y)
    if not getattr(module, "sparse_product", False):
        return torch_kernel.structured(
            module, x, y, get_kernel("cartesian_product", "cuda")
        )
    result = sparse_product(x, y, operands(module))
    return result.index_select(-1, module.indices)


def evaluate(module, x, y, weight):
    return torch_kernel.evaluate(module, x, y, weight)
