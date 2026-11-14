"""Native sparse chains and their transpose, visible to joint AOT compilation."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch import sparse_chain as torch_kernel
from .sparse_axis import evaluate as axis


@triton_op("eqx::sparse_chain", mutates_args=())
def contract(
    features: torch.Tensor, lefts: list[int], tables: list[torch.Tensor], reverse: bool
) -> torch.Tensor:
    order = range(len(lefts) - 1, -1, -1) if reverse else range(len(lefts))
    for index in order:
        offset = 4 * index + 2 * int(reverse)
        features = axis(features, tables[offset], tables[offset + 1], lefts[index], 1)
    return features.clone() if not lefts else features


def setup_context(ctx, inputs, output):
    _, ctx.lefts, tables, ctx.reverse = inputs
    ctx.save_for_backward(*tables)


def backward(ctx, gradient):
    return (
        contract(gradient, ctx.lefts, list(ctx.saved_tensors), not ctx.reverse),
        None,
        [None] * len(ctx.saved_tensors),
        None,
    )


contract.register_autograd(backward, setup_context=setup_context)


def evaluate(features, lefts, reverse, *tables):
    if not lefts:
        return features
    if forward_ad_active() or any(value.requires_grad for value in tables):
        return torch_kernel.evaluate(features, lefts, reverse, *tables)
    return contract(features, list(lefts), list(tables), reverse)


@triton.jit
def chain_kernel(
    features,
    result,
    indices,
    weights,
    rows,
    WIDTH: tl.constexpr,
    OUTERS: tl.constexpr,
    TERMS: tl.constexpr,
    OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, BLOCK)
    valid = (row[:, None] < rows) & (column[None, :] < WIDTH)
    value = tl.load(features + row[:, None] * WIDTH + column[None, :], valid, 0)
    for stage in tl.static_range(len(OUTERS)):
        output = tl.full(
            (ROWS, BLOCK),
            0,
            tl.float64 if features.dtype.element_ty == tl.float64 else tl.float32,
        )
        for term in tl.static_range(TERMS[stage]):
            offset = OFFSETS[stage] + (column % OUTERS[stage]) * TERMS[stage] + term
            source = tl.load(indices + offset, column < WIDTH, 0)
            weight = tl.load(weights + offset, column < WIDTH, 0)
            source = tl.where(
                column < WIDTH, column // OUTERS[stage] * OUTERS[stage] + source, 0
            )
            output += (
                tl.gather(value, tl.broadcast_to(source[None, :], (ROWS, BLOCK)), 1)
                * weight[None, :]
            )
        value = output
    tl.store(result + row[:, None] * WIDTH + column[None, :], value, valid)


@triton_op("eqx::packed_sparse_chain", mutates_args=())
def packed(
    features: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    outers: list[int],
    terms: list[int],
    offsets: list[int],
    reverse: bool,
) -> torch.Tensor:
    """Keep a short square chain in registers, using paired fixed transposes."""
    validate_uniform((features, weights))
    if not features.is_cuda or not features.is_contiguous():
        raise ValueError("Packed sparse chains require contiguous CUDA features.")
    result = torch.empty_like(features)
    if features.numel():
        width = features.shape[-1]
        rows = features.numel() // width
        order = tuple(range(len(outers) - 1, -1, -1) if reverse else range(len(outers)))
        wrap_triton(chain_kernel)[(triton.cdiv(rows, 4),)](
            features,
            result,
            indices,
            weights,
            rows,
            width,
            tuple(outers[k] for k in order),
            tuple(terms[2 * k + int(reverse)] for k in order),
            tuple(offsets[2 * k + int(reverse)] for k in order),
            triton.next_power_of_2(width),
            4,
            num_warps=4,
        )
    return result


def packed_context(ctx, inputs, output):
    _, indices, weights, ctx.outers, ctx.terms, ctx.offsets, ctx.reverse = inputs
    ctx.save_for_backward(indices, weights)


def packed_backward(ctx, gradient):
    indices, weights = ctx.saved_tensors
    return (
        packed(
            gradient.contiguous(),
            indices,
            weights,
            ctx.outers,
            ctx.terms,
            ctx.offsets,
            not ctx.reverse,
        ),
        None,
        None,
        None,
        None,
        None,
        None,
    )


packed.register_autograd(packed_backward, setup_context=packed_context)
