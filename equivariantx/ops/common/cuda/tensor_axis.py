"""Native matrix actions on tensor axes without expanding broadcast operands."""

from math import prod

import torch

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch.tensor_axis import primal
from .pointwise import broadcast_offset, coordinate_offset, launch, scalar_type
from .pointwise import evaluate as pointwise


@torch.library.custom_op("eqx::tensor_axis", mutates_args=(), device_types="cuda")
def contract(
    features: torch.Tensor, matrix: torch.Tensor, left: int = 1, right: int = 1
) -> torch.Tensor:
    """Apply a broadcast matrix without copying its leading dimensions."""
    validate_uniform((features, matrix))
    if features.ndim < 1 or matrix.ndim < 2:
        raise ValueError("Tensor axes require feature vectors and matrices.")
    inner, outer = matrix.shape[-1], matrix.shape[-2]
    if min(left, right, inner, outer) < 1 or features.shape[-1] != left * inner * right:
        raise ValueError("Matrix dimensions do not match the tensor axis.")
    shape = torch.broadcast_shapes(features.shape[:-1], matrix.shape[:-2])
    result = features.new_empty((*shape, left * outer * right))
    if not result.numel():
        return result
    body = f"""
        I row=i/{left * outer * right}, a=(i/{outer * right})%{left};
        int o=(i/{right})%{outer},b=i%{right};
        I xo={broadcast_offset(features, shape, 1)};
        I wo={broadcast_offset(matrix, shape, 2)};
        T value=0;
        for(int k=0;k<{inner};++k)
            value+=p0[xo+((a*{inner}+k)*{right}+b)*{features.stride(-1)}]
                  *p1[wo+o*{matrix.stride(-2)}+k*{matrix.stride(-1)}];
        out[i]=value;
    """
    return pointwise(result, (features, matrix), body)


@contract.register_fake
def fake(features, matrix, left=1, right=1):
    shape = torch.broadcast_shapes(features.shape[:-1], matrix.shape[:-2])
    return features.new_empty((*shape, left * matrix.shape[-2] * right))


def setup_context(ctx, inputs, output):
    features, matrix, ctx.left, ctx.right = inputs
    ctx.feature_size, ctx.matrix_size = features.shape, matrix.shape
    ctx.save_for_backward(
        features if ctx.needs_input_grad[1] else None,
        matrix if ctx.needs_input_grad[0] else None,
    )


def backward(ctx, gradient):
    features, matrix = ctx.saved_tensors
    left, right = ctx.left, ctx.right
    dx = (
        contract(gradient, matrix.transpose(-1, -2), left, right).sum_to_size(
            ctx.feature_size
        )
        if ctx.needs_input_grad[0]
        else None
    )
    dm = (
        matrix_gradient(features, gradient, list(ctx.matrix_size), left, right)
        if ctx.needs_input_grad[1]
        else None
    )
    return dx, dm, None, None


contract.register_autograd(backward, setup_context=setup_context)


@torch.library.custom_op(
    "eqx::tensor_axis_matrix_gradient", mutates_args=(), device_types="cuda"
)
def matrix_gradient(
    features: torch.Tensor,
    gradient: torch.Tensor,
    sizes: list[int],
    left: int,
    right: int,
) -> torch.Tensor:
    """Reduce all channel and broadcast contributions directly into each matrix."""
    validate_uniform((features, gradient))
    if len(sizes) < 2 or min(features.ndim, gradient.ndim) < 1 or min(sizes) < 0:
        raise ValueError("Tensor axes require feature vectors and matrices.")
    outer, inner = sizes[-2:]
    leading = torch.broadcast_shapes(features.shape[:-1], sizes[:-2])
    if (
        min(outer, inner, left, right) < 1
        or features.shape[-1] != left * inner * right
        or gradient.shape != (*leading, left * outer * right)
    ):
        raise ValueError("Inconsistent tensor-axis gradient shape.")
    if not prod(sizes) or not gradient.numel():
        return features.new_zeros(sizes)
    batch = (1,) * (len(leading) + 2 - len(sizes)) + tuple(sizes[:-2])
    reduced = tuple(size if m == 1 else 1 for size, m in zip(leading, batch))
    samples = prod(reduced) * left * right
    count = prod(sizes)
    parts = min((samples + 255) // 256, max(1, 4096 // count), 128)
    result = features.new_empty((*sizes, parts))

    def offsets(value):
        padding = len(leading) + 1 - value.ndim
        strides = (0,) * padding + tuple(
            s if n != 1 else 0 for n, s in zip(value.shape[:-1], value.stride()[:-1])
        )
        return coordinate_offset(batch, strides, "row"), coordinate_offset(
            reduced, strides, "r"
        )

    xb, xr = offsets(features)
    gb, gr = offsets(gradient)
    dtype = scalar_type(features.dtype)
    source = f"""
    using T = {dtype}; using I = long long;
    extern "C" __global__ void run(const T* x, const T* g, T* out, I tasks, I samples, I parts) {{
        int lane=threadIdx.x%32;
        for (I task=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32;task<tasks;
             task+=I(gridDim.x)*blockDim.x/32) {{
            I entry=task/parts,row=entry/{outer * inner};
            I ii=entry%{inner},oo=(entry/{inner})%{outer};
            I xbase={xb},gbase={gb};
            T acc=0;
            for (I k=(task%parts)*32+lane;k<samples;k+=parts*32) {{
                I r=k/{left * right},a=(k/{right})%{left},b=k%{right};
                acc+=x[xbase+{xr}+((a*{inner}+ii)*{right}+b)*{features.stride(-1)}]
                    *g[gbase+{gr}+((a*{outer}+oo)*{right}+b)*{gradient.stride(-1)}];
            }}
            for(int delta=16;delta;delta/=2)
                acc+=__shfl_down_sync(0xffffffff,acc,delta);
            if(!lane) out[task]=acc;
        }}
    }}
    """
    launch(
        source,
        (features, gradient, result),
        (count * parts, samples, parts),
        count * parts * 32,
        threads=128,
    )
    return result.squeeze(-1) if parts == 1 else result.sum(-1)


@matrix_gradient.register_fake
def fake_matrix_gradient(features, gradient, sizes, left, right):
    return features.new_empty(sizes)


def setup_matrix_gradient(ctx, inputs, output):
    features, gradient, _, ctx.left, ctx.right = inputs
    ctx.feature_size, ctx.gradient_size = features.shape, gradient.shape
    ctx.save_for_backward(
        features if ctx.needs_input_grad[1] else None,
        gradient if ctx.needs_input_grad[0] else None,
    )


def matrix_gradient_backward(ctx, matrix):
    features, gradient = ctx.saved_tensors
    left, right = ctx.left, ctx.right
    dx = (
        contract(gradient, matrix.transpose(-1, -2), left, right).sum_to_size(
            ctx.feature_size
        )
        if ctx.needs_input_grad[0]
        else None
    )
    dg = (
        contract(features, matrix, left, right).expand(ctx.gradient_size)
        if ctx.needs_input_grad[1]
        else None
    )
    return dx, dg, None, None, None


matrix_gradient.register_autograd(
    matrix_gradient_backward, setup_context=setup_matrix_gradient
)


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
