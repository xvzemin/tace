"""Compiler-visible local IEEE matrix products with recursive derivatives."""

import math

import torch

from ....utils.autograd import forward_ad_active, nested_forward_ad
from ...precision import validate_uniform
from ..torch.matmul import matmul as torch_matmul
from .pointwise import launch


@torch.library.custom_op("eqx::ieee_bmm", mutates_args=(), device_types="cuda")
def bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply strided FP32 batches with IEEE CUDA arithmetic."""
    validate_uniform((left, right))
    if left.dtype != torch.float32 or left.ndim != 3 or right.ndim != 3:
        raise ValueError("IEEE batched products require FP32 CUDA matrices.")
    if left.shape[0] != right.shape[0] or left.shape[-1] != right.shape[-2]:
        raise ValueError("Batched matrix dimensions do not match.")
    batch, m, k = left.shape
    n = right.shape[-1]
    output = left.new_empty((batch, m, n))
    if not output.numel():
        return output
    source = """
    using I=long long;
    extern "C" __global__ void run(const float* a,const float* b,float* out,
        I batch,I m,I n,I k,I a0,I a1,I a2,I b0,I b1,I b2) {
        __shared__ float av[16][16],bv[16][16];
        I tiles_m=(m+15)/16,tiles_n=(n+15)/16;
        for(I tile=blockIdx.x;tile<batch*tiles_m*tiles_n;tile+=gridDim.x) {
            I z=tile/(tiles_m*tiles_n);
            I row=((tile/tiles_n)%tiles_m)*16+threadIdx.x/16;
            I col=(tile%tiles_n)*16+threadIdx.x%16;
            int tx=threadIdx.x%16,ty=threadIdx.x/16;
            float total=0;
            for(I first=0;first<k;first+=16) {
                av[ty][tx]=(row<m && first+tx<k) ? a[z*a0+row*a1+(first+tx)*a2] : 0;
                bv[ty][tx]=(col<n && first+ty<k) ? b[z*b0+(first+ty)*b1+col*b2] : 0;
                __syncthreads();
                #pragma unroll
                for(int q=0;q<16;++q) total=fmaf(av[ty][q],bv[q][tx],total);
                __syncthreads();
            }
            if(row<m && col<n) out[(z*m+row)*n+col]=total;
            __syncthreads();
        }
    }
    """
    launch(
        source,
        (left, right, output),
        (batch, m, n, k, *left.stride(), *right.stride()),
        batch * ((m + 15) // 16) * ((n + 15) // 16) * 256,
    )
    return output


@bmm.register_fake
def fake(left, right):
    return left.new_empty((left.shape[0], left.shape[1], right.shape[2]))


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
