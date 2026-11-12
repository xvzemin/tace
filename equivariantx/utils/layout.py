"""Shared feature permutations with inverse-permutation derivatives."""

import torch

from ..ops import config
from .autograd import nested_forward_ad


@torch.compiler.allow_in_graph
def permute(features, index, inverse):
    """Permute features with an inverse adjoint and nested forward derivatives."""
    if config.TORCH_REFERENCE or nested_forward_ad():
        return features.index_select(-1, index)
    return Permute.apply(features, index, inverse)


@torch.compiler.allow_in_graph
class Permute(torch.autograd.Function):
    """Permute features and use the inverse permutation for the adjoint."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, index, inverse):
        return features.index_select(-1, index)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, index, inverse = inputs
        ctx.save_for_backward(inverse, index)
        ctx.save_for_forward(index)

    @staticmethod
    def backward(ctx, gradient):
        inverse, index = ctx.saved_tensors
        return permute(gradient, inverse, index), None, None

    @staticmethod
    def jvp(ctx, tangent, index_tangent, inverse_tangent):
        (index,) = ctx.saved_tensors
        return tangent.index_select(-1, index)
