"""Native sparse angular products with factorized channel weights."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.cuda.sparse_bilinear import evaluate as sparse_product
from ...torch.o3 import tensor_product as torch_kernel


def angular(module, x, y, name):
    tables = tuple(getattr(module, f"{name}_sparse_{k}") for k in range(9))
    return sparse_product(x, y, tables)


def evaluate(module, x, y, weight):
    if config.TORCH_REFERENCE or not x.is_cuda:
        return torch_kernel.evaluate(module, x, y, weight)
    if config.FACTORIZED_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return torch_kernel.evaluate(module, x, y, weight)
    return torch_kernel.compact(module, x, y, weight, angular, native=True)


def convolve(
    module, features, attributes, edge_index, weight, num_nodes, radial, network
):
    return torch_kernel.convolve(
        module,
        features,
        attributes,
        edge_index,
        weight,
        num_nodes,
        radial,
        network,
        backend="cuda" if features.is_cuda else "torch",
    )
