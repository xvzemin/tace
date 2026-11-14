"""Native matrix actions on tensor axes without expanding broadcast operands."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch.tensor_axis import TensorAxis, primal


def broadcast_strides(tensor, shape, trailing):
    """Describe leading broadcast coordinates without copying the tensor."""
    padding = len(shape) - (tensor.ndim - trailing)
    return (0,) * padding + tuple(
        0 if statically_known_true(size == 1) else stride
        for size, stride in zip(tensor.shape[:-trailing], tensor.stride()[:-trailing])
    )


def broadcast_arguments(features, matrix, shape):
    """Use scalar runtime sizes/strides in the Inductor Triton ABI."""
    padding = 8 - len(shape)
    return (
        tuple(shape)
        + (1,) * padding
        + broadcast_strides(features, shape, 1)
        + (0,) * padding
        + broadcast_strides(matrix, shape, 2)
        + (0,) * padding
    )


@triton.jit
def row_offset(row, shape, strides):
    """Locate broadcast rows in arbitrary leading storage."""
    offset = tl.full(row.shape, 0, tl.int64)
    for d in tl.static_range(len(shape) - 1, -1, -1):
        offset += (row % shape[d]) * strides[d]
        row = row // shape[d]
    return offset


@triton.jit
def axis_kernel(
    features,
    matrix,
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
    m0,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    m7,
    feature_last,
    matrix_row,
    matrix_column,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner_index = index % (LEFT * OUTER * RIGHT)
    row = index // (LEFT * OUTER * RIGHT)
    a = inner_index // (OUTER * RIGHT)
    o = (inner_index // RIGHT) % OUTER
    b = inner_index % RIGHT
    shape = (d0, d1, d2, d3, d4, d5, d6, d7)
    feature_offset = row_offset(row, shape, (x0, x1, x2, x3, x4, x5, x6, x7))
    matrix_offset = row_offset(row, shape, (m0, m1, m2, m3, m4, m5, m6, m7))
    value = tl.full(
        (BLOCK,), 0, tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32
    )
    for i in range(INNER):
        x = tl.load(
            features + feature_offset + ((a * INNER + i) * RIGHT + b) * feature_last,
            index < size,
            0,
        )
        w = tl.load(
            matrix + matrix_offset + o * matrix_row + i * matrix_column, index < size, 0
        )
        value += x * w
    tl.store(output + index, value, index < size)


@triton_op("eqx::tensor_axis", mutates_args=())
def contract(
    features: torch.Tensor, matrix: torch.Tensor, left: int = 1, right: int = 1
) -> torch.Tensor:
    """Apply a broadcast matrix to a flattened left-by-input-by-right tensor."""
    validate_uniform((features, matrix))
    if not features.is_cuda or features.ndim < 1 or matrix.ndim < 2:
        raise ValueError("Native tensor axes require CUDA features and matrices.")
    if (
        min(left, right, *matrix.shape[-2:]) < 1
        or features.shape[-1] != left * matrix.shape[-1] * right
    ):
        raise ValueError("Matrix dimensions do not match the flattened tensor axis.")
    shape = torch.broadcast_shapes(features.shape[:-1], matrix.shape[:-2])
    if len(shape) > 8:
        return primal(features, matrix, left, right)
    result = features.new_empty((*shape, left * matrix.shape[-2] * right))
    if not statically_known_true(result.numel() == 0):
        wrap_triton(axis_kernel)[(triton.cdiv(result.numel(), 256),)](
            features,
            matrix,
            result,
            result.numel(),
            *broadcast_arguments(features, matrix, shape),
            features.stride(-1),
            matrix.stride(-2),
            matrix.stride(-1),
            left,
            right,
            matrix.shape[-1],
            matrix.shape[-2],
            256,
        )
    return result


def setup_context(ctx, inputs, output):
    features, matrix, ctx.left, ctx.right = inputs
    ctx.save_for_backward(features, matrix)


contract.register_autograd(TensorAxis.backward, setup_context=setup_context)


def evaluate(features, matrix, left=1, right=1):
    """Use the native operator except when forward AD needs its Torch program."""
    if forward_ad_active():
        return primal(features, matrix, left, right)
    return contract(features, matrix, left, right)


def inline():
    """Keep functional transforms exact and optionally expose Torch algebra."""
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    from ... import config

    return (
        config.TENSOR_AXIS_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ) or (forward_ad_active())


def transform(tensor, matrix, rank):
    """Apply native rectangular maps to every Cartesian index."""
    from ..torch.tensor_axis import transform as torch_transform

    return torch_transform(tensor, matrix, rank, primal if inline() else evaluate)
