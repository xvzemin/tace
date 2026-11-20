"""Differentiable fixed cubic interpolation."""

import math

import torch

from ...utils.autograd import forward_ad_active
from ..precision import validate_uniform
from ..registry import get_kernel, resolve_backend


def check_inputs(x, coefficients, step):
    validate_uniform((x, coefficients))
    if x.dtype not in (torch.float32, torch.float64) or coefficients.dtype != x.dtype:
        raise TypeError(
            "Spline coordinates and coefficients must use the same FP32 or FP64 dtype."
        )
    if (
        x.ndim != 1
        or coefficients.ndim != 3
        or coefficients.shape[1] != 4
        or coefficients.shape[0] < 1
        or coefficients.shape[2] < 1
        or coefficients.device != x.device
        or not math.isfinite(step)
        or step <= 0
    ):
        raise ValueError(
            "Expected x[E], coefficients[K, 4, C] on the same device and a positive step."
        )


@torch.library.custom_op(
    "eqx::uniform_cubic_spline", mutates_args=(), device_types="cuda"
)
def spline_cuda(
    x: torch.Tensor, coefficients: torch.Tensor, step: float, order: int
) -> torch.Tensor:
    from .cuda.spline import evaluate

    check_inputs(x, coefficients, step)
    if not 0 <= order <= 3 or coefficients.requires_grad:
        raise ValueError(
            "The CUDA spline requires fixed coefficients and derivative order 0 to 3."
        )
    return evaluate(x, coefficients, step, order)


@spline_cuda.register_fake
def spline_output(x, coefficients, step, order):
    return x.new_empty((x.shape[0], coefficients.shape[2]))


def spline_context(ctx, inputs, output):
    x, coefficients, ctx.step, ctx.order = inputs
    ctx.save_for_backward(x, coefficients)


def spline_backward(ctx, gradient):
    x, coefficients = ctx.saved_tensors
    if ctx.order == 3:
        return x * 0, None, None, None
    dy = spline_cuda(x, coefficients, ctx.step, ctx.order + 1)
    return (gradient * dy).sum(-1), None, None, None


spline_cuda.register_autograd(spline_backward, setup_context=spline_context)


def uniform_cubic_spline(x, coefficients, step, *, backend=None):
    """Evaluate cubic polynomial channels on a uniform grid starting at zero.

    Parameters
    ----------
    x : torch.Tensor
        Coordinates of shape ``(E,)``. Values outside the half-open grid are zero.
    coefficients : torch.Tensor
        Polynomial coefficients of shape ``(intervals, 4, channels)``, highest power first.
    step : float
        Grid spacing.
    backend : {None, "torch", "cuda"}, optional
        Execution backend. Trainable coefficients use PyTorch.

    Examples
    --------
    >>> uniform_cubic_spline(torch.zeros(3), torch.ones(2, 4, 5), 0.5).shape
    torch.Size([3, 5])
    """
    check_inputs(x, coefficients, step)
    if (
        resolve_backend(backend, x.device) == "cuda"
        and x.is_cuda
        and not coefficients.requires_grad
        and not forward_ad_active()
    ):
        return spline_cuda(x, coefficients, step, 0)
    return get_kernel("uniform_cubic_spline", "torch")(x, coefficients, step, 0)
