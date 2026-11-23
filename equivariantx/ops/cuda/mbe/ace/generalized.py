"""Sparse channel-wise multilinear contractions."""

import torch

from ....common.cuda.pointwise import launch, scalar_type


@torch.library.custom_op("eqx::generalized_cg", mutates_args=(), device_types="cuda")
def contract(
    inputs: list[torch.Tensor], tables: list[torch.Tensor], role: int
) -> torch.Tensor:
    """Contract the input slots into the selected output slot."""
    ptr, indices, weights = tables[3 * role : 3 * role + 3]
    n, _, channels = inputs[0].shape
    width = ptr.numel() - 1
    result = inputs[0].new_empty((n, width, channels))
    if not result.numel():
        return result
    scalar = scalar_type(result.dtype)
    arity = len(inputs)
    args = ", ".join(f"const T* x{k}" for k in range(arity))
    strides = ", ".join(f"I n{k}, I d{k}, I c{k}" for k in range(arity))
    term = " * ".join(
        f"x{k}[node*n{k} + I(idx[q*{arity}+{k}])*d{k} + c*c{k}]" for k in range(arity)
    )
    source = f"""
        using T={scalar}; using I=long long;
        extern "C" __global__ void run({args}, const int* ptr, const int* idx,
            const T* w, T* out, I total, I width, I channels, {strides}) {{
            for (I i=I(blockIdx.x)*blockDim.x+threadIdx.x; i<total;
                 i+=I(gridDim.x)*blockDim.x) {{
                I c=i%channels, row=i/channels, node=row/width;
                int a=row%width;
                T sum=0;
                for (int q=ptr[a];q<ptr[a+1];++q) sum += w[q]*({term});
                out[i]=sum;
            }}
        }}
    """
    launch(
        source,
        (*inputs, ptr, indices, weights, result),
        (result.numel(), width, channels, *(s for x in inputs for s in x.stride())),
        result.numel(),
    )
    return result


@contract.register_fake
def fake(inputs, tables, role):
    x = inputs[0]
    return x.new_empty((x.shape[0], tables[3 * role].shape[0] - 1, x.shape[2]))


def setup_context(ctx, inputs, output):
    values, tables, ctx.role = inputs
    ctx.arity = len(values)
    ctx.save_for_backward(*values, *tables)


def backward(ctx, gradient):
    saved = ctx.saved_tensors
    operands = list(saved[: ctx.arity])
    tables = list(saved[ctx.arity :])
    operands.insert(ctx.role, gradient)
    grads = [
        contract(operands[:k] + operands[k + 1 :], tables, k) if needed else None
        for k, needed in zip(
            (i for i in range(len(operands)) if i != ctx.role), ctx.needs_input_grad[0]
        )
    ]
    return grads, [None] * len(tables), None


contract.register_autograd(backward, setup_context=setup_context)
