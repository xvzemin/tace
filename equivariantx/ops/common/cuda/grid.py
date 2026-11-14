"""Fused signed permutations and sheet transforms for Fourier grids."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch import grid as torch_kernel


@triton.jit
def sheet_kernel(
    features,
    input_index,
    output_index,
    input_sign,
    output_sign,
    output,
    size,
    channels_width,
    SHEETS: tl.constexpr,
    WIDTH: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < size
    local = index % channels_width
    if REVERSE:
        destination = tl.load(output_index + local, mask, 0)
        sign = tl.load(output_sign + local, mask, 0)
    else:
        destination = local
        sign = 1
    sheet = (destination // WIDTH) % SHEETS
    base = destination // (SHEETS * WIDTH) * (SHEETS * WIDTH) + destination % WIDTH
    value = tl.full(
        (BLOCK,), 0, tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32
    )
    for k in tl.static_range(SHEETS):
        position = base + k * WIDTH
        if REVERSE:
            source = position
            coefficient = sign
        else:
            source = tl.load(input_index + position, mask, 0)
            coefficient = tl.load(input_sign + position, mask, 0)
        parity = sheet & k
        parity = (parity & 1) ^ (parity >> 1)
        coefficient = coefficient * (1 - 2 * parity)
        value += (
            tl.load(
                features + index // channels_width * channels_width + source, mask, 0
            )
            * coefficient
        )
    tl.store(output + index, value * (SHEETS**-0.5), mask)


@triton_op("eqx::grid_sheet", mutates_args=())
def contract(
    features: torch.Tensor,
    input_index: torch.Tensor,
    output_index: torch.Tensor,
    input_sign: torch.Tensor,
    output_sign: torch.Tensor,
    sheets: int,
    width: int,
    reverse: bool = False,
) -> torch.Tensor:
    validate_uniform((features,))
    if not features.is_cuda or features.ndim < 1 or sheets not in (2, 4) or width < 1:
        raise ValueError(
            "Native grid sheets require CUDA features and two or four sheets."
        )
    size = features.shape[-1]
    tables = input_index, output_index, input_sign, output_sign
    if size % (sheets * width) or any(
        value.ndim != 1 or value.numel() != size or value.device != features.device
        for value in tables
    ):
        raise ValueError("Sheet permutation dimensions do not match the features.")
    features = features.contiguous()
    result = torch.empty_like(features)
    if result.numel():
        wrap_triton(sheet_kernel)[(triton.cdiv(result.numel(), 256),)](
            features,
            input_index,
            output_index,
            input_sign,
            output_sign,
            result,
            result.numel(),
            features.shape[-1],
            sheets,
            width,
            reverse,
            256,
        )
    return result


def setup_context(ctx, inputs, output):
    _, *tables, ctx.sheets, ctx.width, ctx.reverse = inputs
    ctx.save_for_backward(*tables)


def backward(ctx, gradient):
    return (
        contract(gradient, *ctx.saved_tensors, ctx.sheets, ctx.width, not ctx.reverse),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


contract.register_autograd(backward, setup_context=setup_context)


def evaluate(
    features,
    input_index,
    output_index,
    input_sign,
    output_sign,
    sheets,
    width,
    reverse=False,
):
    """Retain the native kernel in joint graphs and exact Torch functional transforms."""
    if forward_ad_active():
        return torch_kernel.evaluate(
            features,
            input_index,
            output_index,
            input_sign,
            output_sign,
            sheets,
            width,
            reverse,
        )
    return contract(
        features,
        input_index,
        output_index,
        input_sign,
        output_sign,
        sheets,
        width,
        reverse,
    )


@triton_op("eqx::grid_linear", mutates_args=())
def linear_contract(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Use the shared native GEMM while retaining a constant-basis adjoint."""
    from .channel_linear import evaluate as channel_linear

    inputs, outputs = matrix.shape[1], matrix.shape[0]
    metadata = repr(
        ("ir_mul", (((1, inputs),), ((1, outputs),), ((0, 0, 0, 1.0),), (), ()))
    )
    value = channel_linear(
        metadata,
        features.reshape(-1, inputs),
        matrix.T.contiguous().flatten(),
        features.new_empty(0),
    )
    return value.reshape(*features.shape[:-1], outputs)


def linear_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[1])


def linear_backward(ctx, gradient):
    return linear_contract(gradient, ctx.saved_tensors[0].T), None


linear_contract.register_autograd(linear_backward, setup_context=linear_context)


def linear(features, matrix):
    if matrix.requires_grad or forward_ad_active():
        return torch_kernel.linear(features, matrix)
    return linear_contract(features, matrix)
