"""Native matrix actions on tensor axes without expanding broadcast operands."""

import torch

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch.tensor_axis import TensorAxis, primal
from .pointwise import broadcast_offset
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
