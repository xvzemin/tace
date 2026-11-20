"""Shared row-local expression execution and compiler-visible lowering."""

import math

import torch

from .compilation import edge_work, should_inline


def silu_derivative(x, factors, scale: float, order: int):
    """Apply a scaled SiLU derivative and its pointwise cotangents."""
    if order == 0:
        out = torch.nn.functional.silu(x)
    else:
        s = torch.sigmoid(x)
        d = s * (1 - s)
        if order == 1:
            out = torch.addcmul(s, x, d)
        elif order == 2:
            out = d * (2 + x * (1 - 2 * s))
        elif order == 3:
            out = d * (3 * (1 - 2 * s) + x * (1 - 6 * d))
        else:
            raise ValueError("Expected a SiLU derivative order from 0 to 3.")
    out = out * scale
    for v in factors:
        out = out * v
    return out


@torch.library.custom_op("eqx::silu_product", mutates_args=(), device_types="cuda")
def silu_product(
    x: torch.Tensor, factors: list[torch.Tensor], scale: float, order: int
) -> torch.Tensor:
    from ..precision import validate_uniform

    validate_uniform((x, factors))
    if not 0 <= order <= 2 or any(f.shape != x.shape for f in factors):
        raise ValueError(
            "Expected equal-shaped factors and a derivative order from 0 to 2."
        )
    return silu_evaluate(x, factors, scale, order)


def silu_evaluate(x, factors, scale, order):
    from .cuda.pointwise import evaluate

    inputs = [x.contiguous(), *(f.contiguous() for f in factors)]
    expression = ("v * s", "s + v * d", "d * (T(2) + v * (T(1) - T(2)*s))")[order]
    for i in range(len(factors)):
        expression = f"({expression}) * p{i + 1}[i]"
    return evaluate(
        torch.empty_like(inputs[0]),
        inputs,
        f"""
        T v = p0[i], s = T(1)/(T(1)+exp(-v)), d = s*(T(1)-s);
        out[i] = T({float(scale).hex()}) * ({expression});
    """,
    )


@silu_product.register_fake
def silu_output(x, factors, scale, order):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def silu_context(ctx, inputs, output):
    x, factors, ctx.scale, ctx.order = inputs
    ctx.save_for_backward(x, *factors)


def silu_backward(ctx, grad):
    x, *factors = ctx.saved_tensors
    next_op = silu_derivative if ctx.order == 2 else silu_product
    dx = (
        next_op(x, [*factors, grad], ctx.scale, ctx.order + 1)
        if ctx.needs_input_grad[0]
        else None
    )
    df = [
        silu_product(x, [*factors[:i], *factors[i + 1 :], grad], ctx.scale, ctx.order)
        if f.requires_grad
        else None
        for i, f in enumerate(factors)
    ]
    return dx, df, None, None


silu_product.register_autograd(silu_backward, setup_context=silu_context)


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
