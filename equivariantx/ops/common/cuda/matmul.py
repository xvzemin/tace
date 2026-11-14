"""Compiler-visible local IEEE matrix products with recursive derivatives."""

import math

import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active, nested_forward_ad
from ...precision import validate_uniform
from ..torch.matmul import matmul as torch_matmul


@triton.jit(do_not_specialize=["M", "N", "K", "A0", "A1", "A2", "B0", "B1", "B2"])
def kernel(
    left,
    right,
    output,
    M,
    N,
    K,
    A0,
    A1,
    A2,
    B0,
    B1,
    B2,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(2)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        k = step * BK + inner
        a = tl.load(
            left + batch * A0 + row[:, None] * A1 + k[None, :] * A2,
            (row[:, None] < M) & (k[None, :] < K),
            0,
        )
        b = tl.load(
            right + batch * B0 + k[:, None] * B1 + col[None, :] * B2,
            (k[:, None] < K) & (col[None, :] < N),
            0,
        )
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    tl.store(
        output + batch * M * N + row[:, None] * N + col[None, :],
        accumulator,
        (row[:, None] < M) & (col[None, :] < N),
    )


@triton_op("eqx::ieee_bmm", mutates_args=())
def bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Apply a local FP32 policy while exposing kernels to Inductor."""
    validate_uniform((left, right))
    if not left.is_cuda or left.dtype != torch.float32:
        raise ValueError("The native IEEE kernel requires FP32 CUDA operands.")
    if (
        left.ndim != 3
        or right.ndim != 3
        or left.shape[0] != right.shape[0]
        or left.shape[-1] != right.shape[-2]
    ):
        raise ValueError("Batched matrix dimensions do not match.")
    batch, m, k = left.shape
    n = right.shape[-1]
    output = left.new_empty((batch, m, n))
    if not statically_known_true(output.numel() == 0):
        wrap_triton(kernel)[(triton.cdiv(m, 32), triton.cdiv(n, 32), batch)](
            left,
            right,
            output,
            m,
            n,
            k,
            *left.stride(),
            *right.stride(),
            32,
            32,
            32,
            num_warps=4,
        )
    return output


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def backward(ctx, gradient):
    left, right = ctx.saved_tensors
    return (
        bmm(gradient, right.transpose(-1, -2)) if ctx.needs_input_grad[0] else None,
        bmm(left.transpose(-1, -2), gradient) if ctx.needs_input_grad[1] else None,
    )


bmm.register_autograd(backward, setup_context=setup_context)


def native_matmul(left, right):
    shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    batch = math.prod(shape)
    m, k = left.shape[-2:]
    n = right.shape[-1]
    a = left.expand(*shape, m, k).reshape(batch, m, k)
    b = right.expand(*shape, k, n).reshape(batch, k, n)
    return bmm(a, b).reshape(*shape, m, n)


class Matmul(torch.autograd.Function):
    """Bilinear forward-AD and batching rules around the same native GEMM."""

    @staticmethod
    def forward(left, right):
        return native_matmul(left, right)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return None, None
        left, right = ctx.saved_tensors
        return (
            Matmul.apply(gradient, right.transpose(-1, -2)).sum_to_size(left.shape)
            if ctx.needs_input_grad[0]
            else None,
            Matmul.apply(left.transpose(-1, -2), gradient).sum_to_size(right.shape)
            if ctx.needs_input_grad[1]
            else None,
        )

    @staticmethod
    def jvp(ctx, left_tangent, right_tangent):
        left, right = ctx.saved_tensors
        terms = []
        if left_tangent is not None:
            terms.append(Matmul.apply(left_tangent, right))
        if right_tangent is not None:
            terms.append(Matmul.apply(left, right_tangent))
        if not terms:
            return (left.sum(-1, keepdim=True) + right.sum(-2, keepdim=True)) * 0
        return terms[0] if len(terms) == 1 else terms[0] + terms[1]

    @staticmethod
    def vmap(info, in_dims, left, right):
        a = left.unsqueeze(0) if in_dims[0] is None else left.movedim(in_dims[0], 0)
        b = right.unsqueeze(0) if in_dims[1] is None else right.movedim(in_dims[1], 0)
        rank = max(a.ndim, b.ndim)
        a = a.reshape(a.shape[0], *([1] * (rank - a.ndim)), *a.shape[1:])
        b = b.reshape(b.shape[0], *([1] * (rank - b.ndim)), *b.shape[1:])
        return Matmul.apply(a, b), 0


def matmul(left, right):
    """Use local IEEE GEMMs and ordinary FP64 or nested-forward-AD arithmetic."""
    validate_uniform((left, right))
    if left.ndim < 2 or right.ndim < 2 or left.shape[-1] != right.shape[-2]:
        raise ValueError("Matrix dimensions do not match.")
    if not left.is_cuda or left.dtype == torch.float64 or nested_forward_ad():
        return torch_matmul(left, right)
    if forward_ad_active():
        return Matmul.apply(left, right)
    return native_matmul(left, right)
