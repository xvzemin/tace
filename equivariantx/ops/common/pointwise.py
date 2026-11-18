"""Shared row-local expression execution and compiler-visible lowering."""

import math

import torch

from .compilation import edge_work, should_inline


def activation_vjp(x, grad, name, order):
    """Apply the first or second derivative of a scalar activation."""
    if name == "tanh":
        t = x.tanh()
        first = grad * (1 - t.square())
        return first if order == 1 else -2 * t * first
    s = x.sigmoid()
    d = s * (1 - s)
    if name == "silu" and order == 1:
        return grad * torch.addcmul(s, x, d)
    first = grad * d
    if name == "sigmoid":
        return first if order == 1 else first * (1 - 2 * s)
    return first * (2 + x * (1 - 2 * s))


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
    if should_inline(inline, edge_work, metadata):
        return torch_evaluate(features, metadata, output_dim)
    batch = math.prod(features.shape[:-1])
    index = torch.zeros(batch, dtype=torch.int64, device=features.device)
    inputs = [features.reshape(batch, features.shape[-1])]
    output = edge_program(metadata, inputs, index, index, batch, inline=False)[0]
    return output.reshape(*features.shape[:-1], output_dim)
