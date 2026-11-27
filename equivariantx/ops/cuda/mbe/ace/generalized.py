"""Sparse channel-wise multilinear contractions."""

from functools import lru_cache
from itertools import accumulate

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
    ctx.save_for_backward(*(values if ctx.arity > 1 else ()), *tables)


def backward(ctx, gradient):
    saved = ctx.saved_tensors
    if ctx.arity == 1:
        return (
            [contract([gradient], list(saved), 1 - ctx.role)],
            [None] * len(saved),
            None,
        )
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


@lru_cache(maxsize=128)
def row_source(dtype, sizes, groups, transpose):
    """Route angular rows between stage tensors and contiguous GEMM blocks."""
    ends = tuple(accumulate(sizes))
    stops = tuple(accumulate(groups))

    def choose(var, bounds, expressions):
        return (
            "".join(
                f"{var}<{end} ? {expr} : "
                for end, expr in zip(bounds[:-1], expressions[:-1])
            )
            + expressions[-1]
        )

    tensors = ",".join(
        f"{'' if transpose else 'const '}T* x{k}" for k in range(len(sizes))
    )
    strides = (
        ",".join(f"I sn{k}, I sr{k}, I sc{k}" for k in range(len(sizes)))
        if not transpose
        else "I sn, I sc"
    )
    packed_row = choose(
        "j",
        stops,
        [
            f"(n*{start}+node*{width}+j-{start})"
            for start, width in zip((0, *stops), groups)
        ],
    )
    if transpose:
        put = choose(
            "r",
            ends,
            [
                f"(x{k}+(node*{width}+r-{start})*channels+c)"
                for k, (start, width) in enumerate(zip((0, *ends), sizes))
            ],
        )
        body = f"""T sum=0;
            for(int q=ptr[r];q<ptr[r+1];++q) {{
                int j=reverse[q];
                sum+=weights[j]*packed[({packed_row})*sn+c*sc];
            }}
            *({put})=sum;"""
        width = ends[-1]
    else:
        read = choose(
            "r",
            ends,
            [
                f"x{k}[node*sn{k}+(r-{start})*sr{k}+c*sc{k}]"
                for k, start in enumerate((0, *ends[:-1]))
            ],
        )
        body = f"int j=r0, r=indices[j]; packed[({packed_row})*channels+c]=({read})*weights[j];"
        width = stops[-1]
    return f"""using I=long long; using T={scalar_type(dtype)};
        extern "C" __global__ void run({tensors},{"const " if transpose else ""}T* packed,
        const int* indices,const T* weights,const int* ptr,const int* reverse,I n,I channels,{strides}) {{
        for(I i=I(blockIdx.x)*blockDim.x+threadIdx.x; i<n*{width}*channels;i+=I(gridDim.x)*blockDim.x) {{
          I c=i%channels, node=i/channels/{width}; int {"r" if transpose else "r0"}=int(i/channels%{width});
          {body}
        }} }}"""


def rows(sizes, groups, tables, values, transpose):
    x = values[0]
    channels = x.shape[-1]
    if transpose:
        (packed,) = values
        n = packed.shape[0] // sum(groups)
        stages = tuple(x.new_empty((n, width, channels)) for width in sizes)
        work, strides = sum(sizes), packed.stride()
    else:
        stages = values
        n = x.shape[0]
        packed = x.new_empty((n * sum(groups), channels))
        work, strides = sum(groups), tuple(v for t in stages for v in t.stride())
    launch(
        row_source(x.dtype, sizes, groups, transpose),
        (*stages, packed, *tables),
        (n, channels, *strides),
        n * work * channels,
    )
    return stages if transpose else packed


class PackRows(torch.autograd.Function):
    """Gather scaled stage rows into the coefficient GEMM layout."""

    @staticmethod
    def forward(ctx, sizes, groups, tables, *stages):
        ctx.sizes, ctx.groups = sizes, groups
        ctx.save_for_backward(*tables)
        return rows(sizes, groups, tables, stages, False)

    @staticmethod
    def backward(ctx, grad):
        return (
            None,
            None,
            None,
            *UnpackRows.apply(ctx.sizes, ctx.groups, ctx.saved_tensors, grad),
        )


class UnpackRows(torch.autograd.Function):
    """Accumulate coefficient-row gradients into their source stage rows."""

    @staticmethod
    def forward(ctx, sizes, groups, tables, packed):
        ctx.sizes, ctx.groups = sizes, groups
        ctx.save_for_backward(*tables)
        return rows(sizes, groups, tables, (packed,), True)

    @staticmethod
    def backward(ctx, *grads):
        return (
            None,
            None,
            None,
            PackRows.apply(ctx.sizes, ctx.groups, ctx.saved_tensors, *grads),
        )
