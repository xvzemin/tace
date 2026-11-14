"""Native bounded sparse contractions without materializing gathered terms."""

import torch

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch.sparse_axis import SparseAxis, primal
from .pointwise import broadcast_offset
from .pointwise import evaluate as pointwise


@torch.library.custom_op("eqx::sparse_axis", mutates_args=(), device_types="cuda")
def contract(
    features: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    left: int = 1,
    right: int = 1,
) -> torch.Tensor:
    """Apply a sparse broadcast matrix without gathered-term intermediates."""
    validate_uniform((features, weights))
    if indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 2:
        raise ValueError("Sparse axes require a two-dimensional integer table.")
    if (
        weights.shape[-2:] != indices.shape
        or min(left, right, *indices.shape) < 1
        or features.shape[-1] % (left * right)
    ):
        raise ValueError("Sparse matrix dimensions do not match the tensor axis.")
    shape = torch.broadcast_shapes(features.shape[:-1], weights.shape[:-2])
    outer, terms = indices.shape
    inner = features.shape[-1] // (left * right)
    result = features.new_empty((*shape, left * outer * right))
    if not result.numel():
        return result
    body = f"""
        I row=i/{left * outer * right},a=(i/{outer * right})%{left};
        int o=(i/{right})%{outer},b=i%{right};
        I xo={broadcast_offset(features, shape, 1)};
        I wo={broadcast_offset(weights, shape, 2)};
        T value=0;
        for(int k=0;k<{terms};++k) {{
            I source=p1[o*{indices.stride(-2)}+k*{indices.stride(-1)}];
            value+=p0[xo+((a*{inner}+source)*{right}+b)*{features.stride(-1)}]
                   *p2[wo+o*{weights.stride(-2)}+k*{weights.stride(-1)}];
        }}
        out[i]=value;
    """
    return pointwise(result, (features, indices, weights), body)


@contract.register_fake
def fake(features, indices, weights, left=1, right=1):
    shape = torch.broadcast_shapes(features.shape[:-1], weights.shape[:-2])
    return features.new_empty((*shape, left * indices.shape[0] * right))


def setup_context(ctx, inputs, output):
    features, indices, weights, ctx.left, ctx.right = inputs
    ctx.save_for_backward(features, indices, weights)


contract.register_autograd(SparseAxis.backward, setup_context=setup_context)


def evaluate(features, indices, weights, left=1, right=1):
    """Preserve both ordinary duals and functional transforms."""
    if forward_ad_active():
        return primal(features, indices, weights, left, right)
    return contract(features, indices, weights, left, right)
