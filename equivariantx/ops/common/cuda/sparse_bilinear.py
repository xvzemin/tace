"""Native fused sparse products with exact recursive transpose kernels."""

import torch

from ...precision import validate_uniform
from ..torch import sparse_bilinear as torch_kernel
from .pointwise import broadcast_offset
from .pointwise import evaluate as pointwise


@torch.library.custom_op("eqx::sparse_bilinear", mutates_args=(), device_types="cuda")
def contract(
    x: torch.Tensor, y: torch.Tensor, tables: list[torch.Tensor], role: int
) -> torch.Tensor:
    """Fuse input gathers, coefficient products and output reductions."""
    if role not in (0, 1, 2) or len(tables) != 9:
        raise ValueError("Sparse bilinear maps require three orientations.")
    left, right, coefficient = tables[3 * role : 3 * role + 3]
    if (
        left.ndim != 2
        or right.shape != left.shape
        or coefficient.shape != left.shape
        or left.shape[1] < 1
    ):
        raise ValueError("Sparse tables must have matching two-dimensional shapes.")
    if left.dtype not in (torch.int32, torch.int64) or right.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Sparse indices must be integers.")
    validate_uniform((x, y, coefficient))
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    left, right, coefficient = (v.contiguous() for v in (left, right, coefficient))
    width, terms = left.shape
    result = x.new_empty((*shape, width))
    if not result.numel():
        return result
    if not x.shape[-1] or not y.shape[-1]:
        return result.zero_()
    body = f"""
        I row=i/{width},c=i%{width};
        I xo={broadcast_offset(x, shape, 1)},yo={broadcast_offset(y, shape, 1)};
        T total=0;
        for(int k=0;k<{terms};++k) {{
            I q=c*{terms}+k;
            total += p0[xo+p2[q]*{x.stride(-1)}]
                    *p1[yo+p3[q]*{y.stride(-1)}]*p4[q];
        }}
        out[i]=total;
    """
    return pointwise(result, (x, y, left, right, coefficient), body)


@contract.register_fake
def fake(x, y, tables, role):
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    return x.new_empty((*shape, tables[3 * role].shape[0]))


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
