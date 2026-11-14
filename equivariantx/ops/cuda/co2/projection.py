"""Native plane polynomials and broadcast tensor-axis projection."""

import math

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ...common.cuda.tensor_axis import broadcast_arguments, inline, row_offset
from ...common.cuda.tensor_axis import evaluate as tensor_axis
from ...precision import validate_uniform
from ...torch.co2 import projection as torch_kernel


@triton.jit
def plane_kernel(direction, output, rows, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(3):
        a = tl.load(direction + row * 3 + i, row < rows, 0)
        for j in tl.static_range(3):
            b = tl.load(direction + row * 3 + j, row < rows, 0)
            tl.store(
                output + row * 9 + i * 3 + j, (1 if i == j else 0) - a * b, row < rows
            )


@triton_op("eqx::plane_projector", mutates_args=())
def plane_operator(direction: torch.Tensor) -> torch.Tensor:
    """Form native I - n n.T matrices in a single launch."""
    validate_uniform((direction,))
    if not direction.is_cuda or direction.ndim < 1 or direction.shape[-1] != 3:
        raise ValueError("Native plane projectors require CUDA vectors of size three.")
    direction = direction.contiguous()
    result = direction.new_empty((*direction.shape[:-1], 3, 3))
    rows = direction.numel() // 3
    if rows:
        wrap_triton(plane_kernel)[(triton.cdiv(rows, 128),)](
            direction, result, rows, 128
        )
    return result


def plane_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def plane_backward(ctx, gradient):
    (direction,) = ctx.saved_tensors
    return -((gradient + gradient.transpose(-1, -2)) * direction.unsqueeze(-2)).sum(-1)


plane_operator.register_autograd(plane_backward, setup_context=plane_context)


@triton.jit
def detrace_kernel(
    features,
    plane,
    output,
    rows,
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
    plane_row,
    plane_column,
    RANK: tl.constexpr,
    COEFFICIENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    shape = (d0, d1, d2, d3, d4, d5, d6, d7)
    x_offset = row_offset(row, shape, (x0, x1, x2, x3, x4, x5, x6, x7))
    p_offset = row_offset(row, shape, (m0, m1, m2, m3, m4, m5, m6, m7))
    if RANK == 2:
        trace = tl.full(
            (BLOCK,),
            0,
            tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32,
        )
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                x = tl.load(
                    features + x_offset + (3 * i + j) * feature_last, row < rows, 0
                )
                a = tl.load(
                    plane + p_offset + i * plane_row + j * plane_column, row < rows, 0
                )
                b = tl.load(
                    plane + p_offset + j * plane_row + i * plane_column, row < rows, 0
                )
                trace += x * ((a + b) * 0.5)
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                x = tl.load(
                    features + x_offset + (3 * i + j) * feature_last, row < rows, 0
                )
                a = tl.load(
                    plane + p_offset + i * plane_row + j * plane_column, row < rows, 0
                )
                b = tl.load(
                    plane + p_offset + j * plane_row + i * plane_column, row < rows, 0
                )
                tl.store(
                    output + row * 9 + i * 3 + j,
                    x + COEFFICIENT * trace * ((a + b) * 0.5),
                    row < rows,
                )
    else:
        traces = ()
        for k in tl.static_range(3):
            trace = tl.full(
                (BLOCK,),
                0,
                tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32,
            )
            for i in tl.static_range(3):
                for j in tl.static_range(3):
                    a = tl.load(
                        plane + p_offset + i * plane_row + j * plane_column,
                        row < rows,
                        0,
                    )
                    b = tl.load(
                        plane + p_offset + j * plane_row + i * plane_column,
                        row < rows,
                        0,
                    )
                    x = tl.load(
                        features + x_offset + (9 * i + 3 * j + k) * feature_last,
                        row < rows,
                        0,
                    )
                    y = tl.load(
                        features + x_offset + (9 * i + 3 * k + j) * feature_last,
                        row < rows,
                        0,
                    )
                    z = tl.load(
                        features + x_offset + (9 * k + 3 * i + j) * feature_last,
                        row < rows,
                        0,
                    )
                    trace += (x + y + z) * ((a + b) * 0.5)
            traces += (trace / 3,)
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                for k in tl.static_range(3):
                    a = tl.load(
                        plane + p_offset + i * plane_row + j * plane_column,
                        row < rows,
                        0,
                    )
                    b = tl.load(
                        plane + p_offset + j * plane_row + i * plane_column,
                        row < rows,
                        0,
                    )
                    c = tl.load(
                        plane + p_offset + i * plane_row + k * plane_column,
                        row < rows,
                        0,
                    )
                    d = tl.load(
                        plane + p_offset + k * plane_row + i * plane_column,
                        row < rows,
                        0,
                    )
                    e = tl.load(
                        plane + p_offset + j * plane_row + k * plane_column,
                        row < rows,
                        0,
                    )
                    f = tl.load(
                        plane + p_offset + k * plane_row + j * plane_column,
                        row < rows,
                        0,
                    )
                    x = tl.load(
                        features + x_offset + (9 * i + 3 * j + k) * feature_last,
                        row < rows,
                        0,
                    )
                    correction = (
                        (a + b) * traces[k] + (c + d) * traces[j] + (e + f) * traces[i]
                    ) / 6
                    tl.store(
                        output + row * 27 + 9 * i + 3 * j + k,
                        x + COEFFICIENT * correction,
                        row < rows,
                    )


def native_detrace(tensor, plane, rank, coefficient):
    """Launch the shared rank-two/three polynomial with broadcast strides."""
    validate_uniform((tensor, plane))
    if not tensor.is_cuda or tensor.shape[-1] != 3**rank or plane.shape[-2:] != (3, 3):
        raise ValueError(
            "Native detracing requires matching CUDA tensors and 3x3 planes."
        )
    shape = torch.broadcast_shapes(tensor.shape[:-1], plane.shape[:-2])
    if len(shape) > 8:
        function = torch_kernel.detrace_two if rank == 2 else torch_kernel.detrace_three
        return function(tensor, plane, coefficient)
    result = tensor.new_empty((*shape, 3**rank))
    rows = math.prod(shape)
    if rows:
        wrap_triton(detrace_kernel)[(triton.cdiv(rows, 128),)](
            tensor,
            plane,
            result,
            rows,
            *broadcast_arguments(tensor, plane, shape),
            tensor.stride(-1),
            plane.stride(-2),
            plane.stride(-1),
            rank,
            coefficient,
            128,
        )
    return result


@triton_op("eqx::planar_detrace_two", mutates_args=())
def detrace_two(
    tensor: torch.Tensor, plane: torch.Tensor, coefficient: float = -0.5
) -> torch.Tensor:
    """Fuse rank-two trace correction without a dense projection matrix."""
    return native_detrace(tensor, plane, 2, coefficient)


@triton_op("eqx::planar_detrace_three", mutates_args=())
def detrace_three(
    tensor: torch.Tensor, plane: torch.Tensor, coefficient: float = -0.75
) -> torch.Tensor:
    """Fuse rank-three trace correction without dense basis contractions."""
    return native_detrace(tensor, plane, 3, coefficient)


def detrace_context(ctx, inputs, output):
    tensor, plane, ctx.coefficient = inputs
    ctx.save_for_backward(tensor, plane)


detrace_two.register_autograd(
    torch_kernel.DetraceTwo.backward, setup_context=detrace_context
)


detrace_three.register_autograd(
    torch_kernel.DetraceThree.backward, setup_context=detrace_context
)


def plane_projector(direction):
    """Select fused plane formation or compiler-visible ordinary algebra."""
    return (
        torch_kernel.plane_projector(direction)
        if inline()
        else plane_operator(direction)
    )


def project(module, tensor, plane):
    """Apply native broadcast contractions along every Cartesian tensor index."""
    if inline() or module.rank == 0:
        return torch_kernel.project(module, tensor, plane)
    for axis in range(module.rank):
        tensor = tensor_axis(tensor, plane, 3**axis, 3 ** (module.rank - axis - 1))
    return tensor


def detrace(module, tensor, plane):
    """Apply the native low-rank polynomial or compact native trace sequence."""
    if inline() or module.rank < 2:
        return torch_kernel.detrace(module, tensor, plane)
    if module.rank == 2:
        return detrace_two(tensor, plane, module.coefficients[1])
    if module.rank == 3:
        return detrace_three(tensor, plane, module.coefficients[1])
    return torch_kernel.compact_detrace(module, tensor, plane, tensor_axis)
