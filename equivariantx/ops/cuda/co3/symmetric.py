"""Native normalized symmetric maps built from reusable tensor-axis kernels."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda import sparse_axis, tensor_axis
from ...common.cuda.pointwise import evaluate as pointwise
from ...common.cuda.pointwise import launch, scalar_type
from ...common.polynomial import native_polynomial
from ...common.symmetric import matrix_inputs, matrix_metadata
from ...common.torch import sparse_axis as torch_sparse
from ...common.torch import tensor_axis as torch_axis
from ...precision import validate_uniform
from ...torch.co3 import symmetric as torch_kernel


def inline():
    """Expose exact Torch algebra for transforms or optional joint fusion."""
    return (
        config.SYMMETRIC_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ) or (forward_ad_active())


def selected():
    return torch_sparse.primal if inline() else sparse_axis.evaluate


def pack(module, tensor):
    return map_tensor(module, tensor, False)


def unpack(module, tensor):
    return map_tensor(module, tensor, True)


def contract(module, tensor, vector):
    if inline():
        return torch_kernel.contract_reference(module, tensor, vector)
    return torch_kernel.sparse_contract(module, tensor, vector, sparse_axis.evaluate)


def trace(module, tensor):
    if inline():
        return torch_kernel.trace_reference(module, tensor)
    return torch_kernel.sparse_trace(module, tensor, sparse_axis.evaluate)


def multiply(module, tensor, matrix):
    if inline() and module.rank > 4:
        return torch_kernel.multiply_reference(module, tensor, matrix)
    return torch_kernel.sparse_multiply(module, tensor, matrix, selected())


def transform(module, tensor, matrix):
    if use_shared_matrix(module, tensor, matrix):
        from ...common.cuda.matmul import matmul

        value = matrix if matrix.ndim == 2 else matrix.squeeze(-3)
        program, types, operands = matrix_inputs(value, module.rank)
        representation = native_polynomial(
            "symmetric_matrix", matrix_metadata(module.rank), program, types, *operands
        )[0].reshape(*value.shape[:-2], module.dim, module.dim)
        return matmul(tensor, representation)
    return torch_kernel.sparse_transform(
        module,
        tensor,
        matrix,
        torch_sparse.evaluate if inline() else sparse_axis.evaluate,
        torch_axis.primal if inline() else tensor_axis.evaluate,
    )


def use_shared_matrix(module, tensor, matrix):
    """Select the optional matrix schedule only for measured channel widths."""
    if (
        not config.SYMMETRIC_MATRIX_NATIVE
        or not tensor.is_cuda
        or tensor.dtype != torch.float32
        or tensor.ndim < 3
        or module.rank not in (4, 6)
        or not statically_known_true(
            tensor.shape[-2] == (128 if module.rank == 4 else 1024)
        )
        or not (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return False
    return matrix.ndim == 2 or (
        matrix.ndim >= 3 and statically_known_true(matrix.shape[-3] == 1)
    )


@torch.library.custom_op("eqx::symmetric_map", mutates_args=(), device_types="cuda")
def map_operator(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    norm: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    terms: int,
    expand: bool,
) -> torch.Tensor:
    """Apply a normalized permutation map and its transpose."""
    validate_uniform((tensor, norm))
    inner = norm.numel() if expand else indices.numel()
    outer = indices.numel() if expand else norm.numel()
    if tensor.shape[-1] != inner:
        raise ValueError("Symmetric map dimensions do not match the input.")
    tensor = tensor.contiguous()
    indices, norm, permutation, offsets = (
        v.contiguous() for v in (indices, norm, permutation, offsets)
    )
    result = tensor.new_empty((*tensor.shape[:-1], outer))
    if not result.numel():
        return result
    if not expand and terms >= 128:
        source = f"""
        using T={scalar_type(tensor.dtype)}; using I=long long;
        extern "C" __global__ void run(
            const T* x,const T* norm,
            const {scalar_type(permutation.dtype)}* permutation,
            const {scalar_type(offsets.dtype)}* offsets,T* out,I count) {{
            int lane=threadIdx.x%32;
            for(I i=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32;
                i<count;i+=I(gridDim.x)*blockDim.x/32) {{
                I row=i/{outer},c=i%{outer};
                T total=0,correction=0;
                for(I k=offsets[c]+lane;k<offsets[c+1];k+=32) {{
                    T value=x[row*{inner}+permutation[k]]-correction;
                    T next=total+value;
                    correction=(next-total)-value;
                    total=next;
                }}
                for(int delta=16;delta>0;delta/=2)
                    total+=__shfl_down_sync(0xffffffff,total,delta);
                if(lane==0) out[i]=total/norm[c];
            }}
        }}
        """
        launch(
            source,
            (tensor, norm, permutation, offsets, result),
            (result.numel(),),
            result.numel() * 32,
        )
        return result
    body = (
        f"I row=i/{outer},c=i%{outer},src=p1[c]; out[i]=p0[row*{inner}+src]/p2[src];"
        if expand
        else f"""
          I row=i/{outer},c=i%{outer}; T total=0;
          for(I k=p4[c];k<p4[c+1];++k) total+=p0[row*{inner}+p3[k]];
          out[i]=total/p2[c];
          """
    )
    return pointwise(result, (tensor, indices, norm, permutation, offsets), body)


@map_operator.register_fake
def map_fake(tensor, indices, norm, permutation, offsets, terms, expand):
    return tensor.new_empty(
        (*tensor.shape[:-1], indices.numel() if expand else norm.numel())
    )


def map_context(ctx, inputs, output):
    tensor, indices, norm, permutation, offsets, ctx.terms, ctx.expand = inputs
    ctx.save_for_backward(
        tensor if ctx.needs_input_grad[2] else tensor.new_empty(0),
        indices,
        norm,
        permutation,
        offsets,
    )


def map_backward(ctx, gradient):
    tensor, indices, norm, permutation, offsets = ctx.saved_tensors
    dx, dn = torch_kernel.map_gradients(
        tensor,
        indices,
        norm,
        gradient,
        ctx.expand,
        ctx.needs_input_grad[2],
        lambda x, expand: map_operator(
            x, indices, norm, permutation, offsets, ctx.terms, expand
        ),
    )
    return dx, None, dn, None, None, None, None


map_operator.register_autograd(map_backward, setup_context=map_context)


def map_tensor(module, tensor, expand):
    """Use a single native map with recursively native mutual adjoints."""
    if module.rank < 2:
        return tensor
    if inline():
        return (torch_kernel.unpack if expand else torch_kernel.pack)(module, tensor)
    terms = math.factorial(module.rank) // math.prod(
        tuple(math.factorial((module.rank + i) // 3) for i in range(3))
    )
    return map_operator(
        tensor,
        module.indices,
        module.norm,
        module.permutation,
        module.offsets,
        terms,
        expand,
    )
