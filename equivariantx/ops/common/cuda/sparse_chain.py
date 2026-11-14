"""Native sparse chains and their transpose, visible to joint AOT compilation."""

import torch

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch import sparse_chain as torch_kernel
from .pointwise import launch, scalar_type
from .sparse_axis import evaluate as axis


@torch.library.custom_op("eqx::sparse_chain", mutates_args=(), device_types="cuda")
def contract(
    features: torch.Tensor, lefts: list[int], tables: list[torch.Tensor], reverse: bool
) -> torch.Tensor:
    order = range(len(lefts) - 1, -1, -1) if reverse else range(len(lefts))
    for index in order:
        offset = 4 * index + 2 * int(reverse)
        features = axis(features, tables[offset], tables[offset + 1], lefts[index], 1)
    return features.clone() if not lefts else features


@contract.register_fake
def fake(features, lefts, tables, reverse):
    width = features.shape[-1]
    for index in range(len(lefts) - 1, -1, -1) if reverse else range(len(lefts)):
        width = lefts[index] * tables[4 * index + 2 * int(reverse)].shape[0]
    return features.new_empty((*features.shape[:-1], width))


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


@torch.library.custom_op(
    "eqx::packed_sparse_chain", mutates_args=(), device_types="cuda"
)
def packed(
    features: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    outers: list[int],
    terms: list[int],
    offsets: list[int],
    reverse: bool,
) -> torch.Tensor:
    """Evaluate a square sparse chain with two shared-memory row buffers."""
    validate_uniform((features, weights))
    if not features.is_contiguous():
        raise ValueError("Packed sparse chains require contiguous features.")
    result = torch.empty_like(features)
    if not result.numel():
        return result
    indices, weights = indices.contiguous(), weights.contiguous()
    width = features.shape[-1]
    rows = features.numel() // width
    order = range(len(outers) - 1, -1, -1) if reverse else range(len(outers))
    stages = []
    for k in order:
        outer, term, offset = (
            outers[k],
            terms[2 * k + int(reverse)],
            offsets[2 * k + int(reverse)],
        )
        stages.append(f"""
            for(int c=threadIdx.x;c<{width};c+=blockDim.x) {{
                T sum=0;
                for(int q=0;q<{term};++q) {{
                    I entry={offset}+(c%{outer})*{term}+q;
                    sum+=a[(c/{outer})*{outer}+indices[entry]]*weights[entry];
                }}
                b[c]=sum;
            }}
            __syncthreads();
            {{ T* tmp=a; a=b; b=tmp; }}
        """)
    source = f"""
        using T={scalar_type(features.dtype)}; using I=long long;
        extern "C" __global__ void run(const T* x,const {scalar_type(indices.dtype)}* indices,const T* weights,T* out,I rows) {{
            extern __shared__ T storage[];
            for(I row=blockIdx.x;row<rows;row+=gridDim.x) {{
                T* a=storage; T* b=storage+{width};
                for(int c=threadIdx.x;c<{width};c+=blockDim.x) a[c]=x[row*{width}+c];
                __syncthreads();
                {"".join(stages)}
                for(int c=threadIdx.x;c<{width};c+=blockDim.x) out[row*{width}+c]=a[c];
                __syncthreads();
            }}
        }}
    """
    if 2 * width * features.element_size() > 48 * 1024:
        value = features
        for k in range(len(outers) - 1, -1, -1) if reverse else range(len(outers)):
            outer, term, offset = (
                outers[k],
                terms[2 * k + int(reverse)],
                offsets[2 * k + int(reverse)],
            )
            value = axis(
                value,
                indices[offset : offset + outer * term].reshape(outer, term),
                weights[offset : offset + outer * term].reshape(outer, term),
                width // outer,
                1,
            )
        return value.clone() if not outers else value
    launch(
        source,
        (features, indices, weights, result),
        (rows,),
        rows * 256,
        shared=2 * width * features.element_size(),
    )
    return result


@packed.register_fake
def packed_fake(features, indices, weights, outers, terms, offsets, reverse):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


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
