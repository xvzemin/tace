"""Fixed Fourier expansions on a finite radial interval."""

import math

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ...utils.autograd import forward_ad_active
from ..precision import validate_uniform
from ..registry import get_kernel, resolve_backend


def check_inputs(x, frequencies, coefficients, num_cos, extent, indices):
    validate_uniform((x, frequencies, coefficients))
    if not all(t.is_floating_point() for t in (x, frequencies, coefficients)):
        raise TypeError("Fourier operands must use FP32 or FP64.")
    if (
        x.ndim != 1
        or frequencies.ndim != 2
        or coefficients.ndim != (2 if indices is None else 3)
        or frequencies.shape != (2, coefficients.shape[-1])
        or min(coefficients.shape) < 1
        or not 0 <= num_cos <= coefficients.shape[-2]
        or not math.isfinite(extent)
        or extent <= 0
    ):
        raise ValueError("Invalid Fourier expansion shape or interval.")
    if indices is not None and (
        indices.shape != x.shape
        or indices.dtype != torch.long
        or indices.device != x.device
    ):
        raise ValueError(
            "indices must be one int64 table index per coordinate on the same device."
        )


def fourier_torch(x, frequencies, coefficients, num_cos, extent, order=0, indices=None):
    inside = (x >= 0) & (x < extent)
    xx = torch.where(inside, x, 0.0).unsqueeze(-1)
    terms = []
    if indices is not None and coefficients.shape[0] == 1:
        coefficients, indices = coefficients[0], None
    grouping = None
    if indices is not None and statically_known_true(
        x.shape[0] >= 32 * coefficients.shape[0]
    ):
        from .torch.grouped import element_grouping

        grouping = element_grouping(indices, coefficients.shape[0], 1, 32)
    for family, rows in enumerate(
        (coefficients[..., :num_cos, :], coefficients[..., num_cos:, :])
    ):
        if rows.shape[-2]:
            w = frequencies[family]
            phase = xx * w
            # cos^(k) and sin^(k), without adding an inexact pi/2 to the phase.
            sine = (family + order) % 2
            v = phase.sin() if sine else phase.cos()
            if (order + (1 - family)) % 4 >= 2:
                v = -v
            if order:
                v = v * w.pow(order)
            if indices is None:
                value = v @ rows.T
            elif grouping is None:
                value = torch.bmm(rows[indices], v.unsqueeze(-1)).squeeze(-1)
            else:
                slots, owners, blocks = grouping
                packed = v.new_zeros((blocks * 32, v.shape[-1])).index_copy(0, slots, v)
                value = (
                    packed.view(blocks, 32, v.shape[-1]) @ rows[owners].mT
                ).flatten(0, 1)[slots]
            terms.append(value)
    return torch.where(inside[:, None], torch.cat(terms, -1), 0.0)


@torch.library.custom_op("eqx::fourier_radial", mutates_args=(), device_types="cuda")
def fourier_cuda(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    coefficients: torch.Tensor,
    num_cos: int,
    extent: float,
    order: int,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    from .cuda.fourier import evaluate

    check_inputs(x, frequencies, coefficients, num_cos, extent, indices)
    if order < 0 or frequencies.requires_grad or coefficients.requires_grad:
        raise ValueError(
            "The CUDA Fourier kernel requires fixed modes and a nonnegative derivative order."
        )
    return evaluate(x, frequencies, coefficients, num_cos, extent, order, indices)


@fourier_cuda.register_fake
def fourier_shape(x, frequencies, coefficients, num_cos, extent, order, indices=None):
    return x.new_empty((x.shape[0], coefficients.shape[-2]))


def fourier_context(ctx, inputs, output):
    x, frequencies, coefficients, ctx.num_cos, ctx.extent, ctx.order, indices = inputs
    ctx.save_for_backward(x, frequencies, coefficients, indices)


def fourier_backward(ctx, gradient):
    x, frequencies, coefficients, indices = ctx.saved_tensors
    slope = fourier_cuda(
        x, frequencies, coefficients, ctx.num_cos, ctx.extent, ctx.order + 1, indices
    )
    return (gradient * slope).sum(-1), None, None, None, None, None, None


fourier_cuda.register_autograd(fourier_backward, setup_context=fourier_context)


def fourier_radial(
    x, frequencies, coefficients, num_cos, extent, *, indices=None, backend=None
):
    """Evaluate cosine and sine expansions, returning zero outside the interval.

    Parameters
    ----------
    x : torch.Tensor
        Coordinates of shape ``(edges,)``.
    frequencies : torch.Tensor
        Cosine and sine frequencies of shape ``(2, terms)``.
    coefficients : torch.Tensor
        Expansion coefficients of shape ``(channels, terms)``, or
        ``(tables, channels, terms)`` with ``indices``.
    num_cos : int
        Number of leading channels using cosine instead of sine.
    extent : float
        Upper endpoint of the half-open interval starting at zero.
    indices : torch.Tensor, optional
        One coefficient-table index in ``[0, tables)`` per coordinate.
    backend : {None, "torch", "cuda"}, optional
        Execution backend. Trainable coefficients or frequencies use PyTorch.

    Examples
    --------
    >>> fourier_radial(torch.zeros(2), torch.ones(2, 3), torch.ones(4, 3), 1, 10.).shape
    torch.Size([2, 4])
    """
    check_inputs(x, frequencies, coefficients, num_cos, extent, indices)
    if (
        resolve_backend(backend, x.device) == "cuda"
        and x.is_cuda
        and not (
            frequencies.requires_grad
            or coefficients.requires_grad
            or forward_ad_active()
        )
    ):
        return fourier_cuda(
            x, frequencies, coefficients, num_cos, float(extent), 0, indices
        )
    return get_kernel("fourier_radial", "torch")(
        x, frequencies, coefficients, num_cos, extent, 0, indices
    )
