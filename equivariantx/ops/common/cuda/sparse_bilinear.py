"""Native fused sparse products with exact recursive transpose kernels."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...precision import validate_uniform
from ..torch import sparse_bilinear as torch_kernel
from .tensor_axis import broadcast_strides, row_offset


@triton.jit
def product_kernel(
    x,
    y,
    left,
    right,
    coefficient,
    output,
    size,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    x_last,
    y_last,
    OUTPUT: tl.constexpr,
    TERMS: tl.constexpr,
    BLOCK: tl.constexpr,
    TERM_BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, component = index // OUTPUT, index % OUTPUT
    shape = (d0, d1, d2, d3, d4, d5, d6, d7)
    xo = row_offset(row, shape, (x0, x1, x2, x3, x4, x5, x6, x7))
    yo = row_offset(row, shape, (y0, y1, y2, y3, y4, y5, y6, y7))
    if TERMS <= 8:
        value = tl.full(
            (BLOCK,),
            0,
            tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32,
        )
        for k in tl.static_range(TERMS):
            position = component * TERMS + k
            a = tl.load(left + position, index < size, 0)
            b = tl.load(right + position, index < size, 0)
            c = tl.load(coefficient + position, index < size, 0)
            u = tl.load(x + xo + a * x_last, index < size, 0)
            v = tl.load(y + yo + b * y_last, index < size, 0)
            value += u * v * c
    else:
        term = tl.arange(0, TERM_BLOCK)
        mask = (index[:, None] < size) & (term[None, :] < TERMS)
        position = component[:, None] * TERMS + term[None, :]
        a = tl.load(left + position, mask, 0)
        b = tl.load(right + position, mask, 0)
        c = tl.load(coefficient + position, mask, 0)
        u = tl.load(x + xo[:, None] + a * x_last, mask, 0)
        v = tl.load(y + yo[:, None] + b * y_last, mask, 0)
        value = tl.sum(u * v * c, axis=1)
    tl.store(output + index, value, index < size)


@triton_op("eqx::sparse_bilinear", mutates_args=())
def contract(
    x: torch.Tensor, y: torch.Tensor, tables: list[torch.Tensor], role: int
) -> torch.Tensor:
    """Fuse input gathers, coefficient products and their output reduction."""
    if role not in (0, 1, 2) or len(tables) != 9:
        raise ValueError(
            "Sparse bilinear maps require three orientations and role 0, 1 or 2."
        )
    left, right, coefficient = tables[3 * role : 3 * role + 3]
    if (
        left.ndim != 2
        or right.shape != left.shape
        or coefficient.shape != left.shape
        or left.shape[-1] < 1
        or left.dtype not in (torch.int32, torch.int64)
        or right.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Sparse bilinear indices and coefficients must have matching two-dimensional shapes."
        )
    validate_uniform((x, y, coefficient))
    if not x.is_cuda or any(
        value.device != x.device for value in (left, right, coefficient)
    ):
        raise ValueError(
            "Native sparse bilinear maps require all tensors on the same CUDA device."
        )
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    if len(shape) > 8 or not x.shape[-1] or not y.shape[-1]:
        return torch_kernel.primal(x, y, tables, role)
    result = x.new_empty((*shape, left.shape[0]))
    if result.numel():
        padding = 8 - len(shape)
        arguments = (
            tuple(shape)
            + (1,) * padding
            + broadcast_strides(x, shape, 1)
            + (0,) * padding
            + broadcast_strides(y, shape, 1)
            + (0,) * padding
        )
        block = 256 if left.shape[1] <= 8 else 32
        wrap_triton(product_kernel)[(triton.cdiv(result.numel(), block),)](
            x,
            y,
            left.contiguous(),
            right.contiguous(),
            coefficient.contiguous(),
            result,
            result.numel(),
            *arguments,
            x.stride(-1),
            y.stride(-1),
            left.shape[0],
            left.shape[1],
            block,
            triton.next_power_of_2(left.shape[1]),
        )
    return result


def setup_context(ctx, inputs, output):
    x, y, tables, ctx.role = inputs
    ctx.save_for_backward(x, y, *tables)


def backward(ctx, gradient):
    x, y, *tables = ctx.saved_tensors
    dx, dy = torch_kernel.adjoints(
        contract, x, y, gradient, tables, ctx.role, ctx.needs_input_grad[:2]
    )
    return dx, dy, [None] * len(tables), None


contract.register_autograd(backward, setup_context=setup_context)


def evaluate(x, y, tables, role=2):
    if (
        torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
        or any(value.requires_grad for value in tables)
    ):
        return torch_kernel.evaluate(x, y, tables, role)
    return contract(x, y, list(tables), role)
