"""Feature permutations with inverse-permutation derivatives."""

import torch


@torch.compiler.allow_in_graph
class _Permute(torch.autograd.Function):
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
        return _Permute.apply(gradient, inverse, index), None, None

    @staticmethod
    def jvp(ctx, tangent, index_tangent, inverse_tangent):
        (index,) = ctx.saved_tensors
        return tangent.index_select(-1, index)
