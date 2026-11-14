"""Constant sparse linear chains with exact recursively transposed derivatives."""

import torch

from ....utils.autograd import forward_ad_active
from .sparse_axis import primal as sparse_axis


def primal(features, lefts, reverse, *tables):
    """Apply local maps while keeping leading Cartesian axes implicit."""
    order = range(len(lefts) - 1, -1, -1) if reverse else range(len(lefts))
    for index in order:
        offset = 4 * index + 2 * int(reverse)
        features = sparse_axis(
            features, tables[offset], tables[offset + 1], lefts[index], 1
        )
    return features


@torch.compiler.allow_in_graph
class SparseChain(torch.autograd.Function):
    """Save fixed coefficients, with no feature tape for the linear maps."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, lefts, reverse, *tables):
        return primal(features, lefts, reverse, *tables)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.lefts, ctx.reverse, *tables = inputs
        ctx.save_for_backward(*tables)
        ctx.save_for_forward(*tables)
        ctx.shape, ctx.dtype, ctx.device = output.shape, output.dtype, output.device

    @staticmethod
    def backward(ctx, gradient):
        tables = ctx.saved_tensors
        value = evaluate(gradient, ctx.lefts, not ctx.reverse, *tables)
        return (value, None, None) + (None,) * len(tables)

    @staticmethod
    def jvp(ctx, gradient, unused_lefts, unused_reverse, *unused_tables):
        if gradient is None:
            return torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        return primal(gradient, ctx.lefts, ctx.reverse, *ctx.saved_tensors)


@torch.compiler.allow_in_graph
def evaluate(features, lefts, reverse, *tables):
    """Use paired exact adjoints; differentiable coefficients use ordinary AD."""
    if not lefts:
        return features
    if forward_ad_active() or any(value.requires_grad for value in tables):
        return primal(features, lefts, reverse, *tables)
    return SparseChain.apply(features, lefts, reverse, *tables)
