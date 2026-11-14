"""Shared row-local expression execution and compiler-visible lowering."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def torch_evaluate(features, metadata, output_dim):
    """Expose row-local algebra and reduction adjoints to the Torch compiler."""
    from .torch.edge import evaluate_torch

    batch = math.prod(features.shape[:-1])
    index = torch.zeros(batch, dtype=torch.int64, device=features.device)
    inputs = [features.reshape(batch, features.shape[-1])]
    output = evaluate_torch(metadata, inputs, index, index, batch, optimized=True)[0]
    return output.reshape(*features.shape[:-1], output_dim)


def evaluate(features, metadata, output_dim, *, inline=True):
    """Evaluate fused row expressions without materializing internal arrays."""
    from .edge import evaluate as edge_program

    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Pointwise kernels require CUDA float32 or float64 features.")
    if inline and (torch.compiler.is_compiling() or get_proxy_mode() is not None):
        return torch_evaluate(features, metadata, output_dim)
    batch = math.prod(features.shape[:-1])
    index = torch.zeros(batch, dtype=torch.int64, device=features.device)
    inputs = [features.reshape(batch, features.shape[-1])]
    output = edge_program(metadata, inputs, index, index, batch)[0]
    return output.reshape(*features.shape[:-1], output_dim)
