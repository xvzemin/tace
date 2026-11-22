"""Fixed Fourier expansions on a finite radial interval."""

import math

import torch

from ...utils.autograd import forward_ad_active
from ..precision import validate_uniform
from ..registry import get_kernel, resolve_backend


def check_inputs(x, frequencies, coefficients, num_cos, extent):
    validate_uniform((x, frequencies, coefficients))
    if not all(t.is_floating_point() for t in (x, frequencies, coefficients)):
        raise TypeError("Fourier operands must use FP32 or FP64.")
    if (
        x.ndim != 1
        or frequencies.ndim != 2
        or coefficients.ndim != 2
        or frequencies.shape != (2, coefficients.shape[1])
        or min(coefficients.shape) < 1
        or not 0 <= num_cos <= coefficients.shape[0]
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("Invalid Fourier expansion shape or interval.")


def fourier_torch(x, frequencies, coefficients, num_cos, extent, order=0):
    inside = (x >= 0) & (x < extent)
    xx = torch.where(inside, x, 0.0).unsqueeze(-1)
    terms = []
    for family, rows in enumerate((coefficients[:num_cos], coefficients[num_cos:])):
        if rows.shape[0]:
            w = frequencies[family]
            phase = xx * w
            # cos^(k) and sin^(k), without adding an inexact pi/2 to the phase.
            sine = (family + order) % 2
            v = phase.sin() if sine else phase.cos()
            if (order + (1 - family)) % 4 >= 2:
                v = -v
            if order:
                v = v * w.pow(order)
            terms.append(v @ rows.T)
    return torch.where(inside[:, None], torch.cat(terms, -1), 0.0)


@torch.library.custom_op("eqx::fourier_radial", mutates_args=(), device_types="cuda")
def fourier_cuda(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    coefficients: torch.Tensor,
    num_cos: int,
    extent: float,
    order: int,
) -> torch.Tensor:
    from .cuda.fourier import evaluate

    check_inputs(x, frequencies, coefficients, num_cos, extent)
    if order < 0 or frequencies.requires_grad or coefficients.requires_grad:
        raise ValueError(
            "The CUDA Fourier kernel requires fixed modes and a nonnegative derivative order."
        )
    return evaluate(x, frequencies, coefficients, num_cos, extent, order)


@fourier_cuda.register_fake
def fourier_shape(x, frequencies, coefficients, num_cos, extent, order):
    return x.new_empty((x.shape[0], coefficients.shape[0]))


def fourier_context(ctx, inputs, output):
    x, frequencies, coefficients, ctx.num_cos, ctx.extent, ctx.order = inputs
    ctx.save_for_backward(x, frequencies, coefficients)


def fourier_backward(ctx, gradient):
    x, frequencies, coefficients = ctx.saved_tensors
    slope = fourier_cuda(
        x, frequencies, coefficients, ctx.num_cos, ctx.extent, ctx.order + 1
    )
    return (gradient * slope).sum(-1), None, None, None, None, None


fourier_cuda.register_autograd(fourier_backward, setup_context=fourier_context)


def fourier_radial(x, frequencies, coefficients, num_cos, extent, *, backend=None):
    """Evaluate cosine and sine expansions, returning zero outside the interval.

    Parameters
    ----------
    x : torch.Tensor
        Coordinates of shape ``(edges,)``.
    frequencies : torch.Tensor
        Cosine and sine frequencies of shape ``(2, terms)``.
    coefficients : torch.Tensor
        Expansion coefficients of shape ``(channels, terms)``.
    num_cos : int
        Number of leading channels using cosine instead of sine.
    extent : float
        Upper endpoint of the half-open interval starting at zero.
    backend : {None, "torch", "cuda"}, optional
        Execution backend. Trainable coefficients or frequencies use PyTorch.

    Examples
    --------
    >>> fourier_radial(torch.zeros(2), torch.ones(2, 3), torch.ones(4, 3), 1, 10.).shape
    torch.Size([2, 4])
    """
    check_inputs(x, frequencies, coefficients, num_cos, extent)
    if (
        resolve_backend(backend, x.device) == "cuda"
        and x.is_cuda
        and not (
            frequencies.requires_grad
            or coefficients.requires_grad
            or forward_ad_active()
        )
    ):
        return fourier_cuda(x, frequencies, coefficients, num_cos, float(extent), 0)
    return get_kernel("fourier_radial", "torch")(
        x, frequencies, coefficients, num_cos, extent
    )
