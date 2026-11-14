"""Native bounded sparse contractions without materializing gathered terms."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch.sparse_axis import SparseAxis, primal
from .tensor_axis import broadcast_arguments, row_offset


@triton.jit
def sparse_kernel(
    features,
    indices,
    weights,
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
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    w6,
    w7,
    feature_last,
    weight_row,
    weight_column,
    index_row,
    index_column,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    TERMS: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    item = index % (LEFT * OUTER * RIGHT)
    row = index // (LEFT * OUTER * RIGHT)
    a, o, b = item // (OUTER * RIGHT), (item // RIGHT) % OUTER, item % RIGHT
    shape = (d0, d1, d2, d3, d4, d5, d6, d7)
    x_offset = row_offset(row, shape, (x0, x1, x2, x3, x4, x5, x6, x7))
    w_offset = row_offset(row, shape, (w0, w1, w2, w3, w4, w5, w6, w7))
    result = tl.full(
        (BLOCK,), 0, tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32
    )
    for k in range(TERMS):
        source = tl.load(indices + o * index_row + k * index_column, index < size, 0)
        value = tl.load(
            features + x_offset + ((a * INNER + source) * RIGHT + b) * feature_last,
            index < size,
            0,
        )
        weight = tl.load(
            weights + w_offset + o * weight_row + k * weight_column, index < size, 0
        )
        result += value * weight
    tl.store(output + index, result, index < size)


@triton_op("eqx::sparse_axis", mutates_args=())
def contract(
    features: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    left: int = 1,
    right: int = 1,
) -> torch.Tensor:
    """Apply a sparse broadcast matrix to a flattened tensor axis."""
    validate_uniform((features, weights))
    if (
        not features.is_cuda
        or indices.device != features.device
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Native sparse axes require CUDA features and integer indices on the same device."
        )
    if (
        indices.ndim != 2
        or weights.shape[-2:] != indices.shape
        or min(left, right, *indices.shape) < 1
        or features.shape[-1] % (left * right)
    ):
        raise ValueError("Sparse matrix dimensions do not match the tensor axis.")
    shape = torch.broadcast_shapes(features.shape[:-1], weights.shape[:-2])
    if len(shape) > 8:
        return primal(features, indices, weights, left, right)
    result = features.new_empty((*shape, left * indices.shape[0] * right))
    if not statically_known_true(result.numel() == 0):
        wrap_triton(sparse_kernel)[(triton.cdiv(result.numel(), 256),)](
            features,
            indices,
            weights,
            result,
            result.numel(),
            *broadcast_arguments(features, weights, shape),
            features.stride(-1),
            weights.stride(-2),
            weights.stride(-1),
            indices.stride(-2),
            indices.stride(-1),
            features.shape[-1] // (left * right),
            indices.shape[0],
            indices.shape[1],
            left,
            right,
            256,
        )
    return result


def setup_context(ctx, inputs, output):
    features, indices, weights, ctx.left, ctx.right = inputs
    ctx.save_for_backward(features, indices, weights)


contract.register_autograd(SparseAxis.backward, setup_context=setup_context)


def evaluate(features, indices, weights, left=1, right=1):
    """Preserve both ordinary duals and functional transforms."""
    if forward_ad_active():
        return primal(features, indices, weights, left, right)
    return contract(features, indices, weights, left, right)
