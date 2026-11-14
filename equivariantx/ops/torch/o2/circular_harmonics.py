"""Circular powers with an exact recursive analytic adjoint."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ....utils.autograd import nested_forward_ad


def reference(vectors, mmax):
    """Evaluate complex powers with ordinary Torch recurrence and AD."""
    real = vectors[..., 0]
    imaginary = vectors[..., 1]
    outputs = [torch.ones_like(real).unsqueeze(-1)]
    order_real = torch.ones_like(real)
    order_imaginary = torch.zeros_like(imaginary)
    for _ in range(1, mmax + 1):
        next_real = order_real * real - order_imaginary * imaginary
        next_imaginary = order_real * imaginary + order_imaginary * real
        order_real = next_real
        order_imaginary = next_imaginary
        outputs.append(torch.stack((order_real, order_imaginary), dim=-1))
    return torch.cat(outputs, dim=-1)


def previous_orders(vectors, mmax):
    """Return real and imaginary powers zero through mmax minus one."""
    previous = Circular.apply(vectors, mmax - 1)
    real = torch.cat((previous[..., :1], previous[..., 1::2]), -1)
    imaginary = torch.cat(
        (torch.zeros_like(previous[..., :1]), previous[..., 2::2]), -1
    )
    return real, imaginary


class Circular(torch.autograd.Function):
    """Retain only the input vector; differentiate z**m recursively."""

    generate_vmap_rule = True

    @staticmethod
    def forward(vectors, mmax):
        return reference(vectors, mmax)

    @staticmethod
    def setup_context(ctx, inputs, output):
        vectors, ctx.mmax = inputs
        ctx.save_for_backward(vectors)
        ctx.save_for_forward(vectors)

    @staticmethod
    def backward(ctx, gradient):
        (vectors,) = ctx.saved_tensors
        if ctx.mmax == 0:
            return vectors * 0, None
        real, imaginary = previous_orders(vectors, ctx.mmax)
        orders = torch.arange(
            1, ctx.mmax + 1, dtype=vectors.dtype, device=vectors.device
        )
        values = gradient[..., 1:].unflatten(-1, (ctx.mmax, 2)) * orders[:, None]
        a, b = values[..., 0], values[..., 1]
        return torch.stack(
            ((a * real + b * imaginary).sum(-1), (b * real - a * imaginary).sum(-1)), -1
        ), None

    @staticmethod
    def jvp(ctx, tangent, unused):
        (vectors,) = ctx.saved_tensors
        if ctx.mmax == 0:
            return vectors[..., :1] * 0
        real, imaginary = previous_orders(vectors, ctx.mmax)
        orders = torch.arange(
            1, ctx.mmax + 1, dtype=vectors.dtype, device=vectors.device
        )
        dx, dy = tangent[..., :1], tangent[..., 1:]
        value = torch.stack(
            (dx * real - dy * imaginary, dx * imaginary + dy * real), -1
        )
        return torch.cat(
            (vectors[..., :1] * 0, (value * orders[:, None]).flatten(-2)), -1
        )


def evaluate(vectors, mmax, normalize):
    """Normalize once and select exact hand-written or ordinary Torch AD."""
    if normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    if (
        mmax < 8
        or torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or not (vectors.requires_grad or torch._C._are_functorch_transforms_active())
        or config.TORCH_REFERENCE
        or not config.TORCH_CIRCULAR_ADJOINT
        or nested_forward_ad()
    ):
        return reference(vectors, mmax)
    return Circular.apply(vectors, mmax)
