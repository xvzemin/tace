"""Sparse matrix actions on tensor axes with exact broadcast adjoints."""

import torch

from ....utils.autograd import nested_forward_ad, torch_gradient_mask


def reference(features, indices, weights, left=1, right=1):
    """Gather every sparse term and reduce with ordinary Torch operations."""
    inner = features.shape[-1] // (left * right)
    value = features.unflatten(-1, (left, inner, right))
    return (value[..., indices, :] * weights[..., None, :, :, None]).sum(-2).flatten(-3)


def primal(features, indices, weights, left=1, right=1):
    """Expose bounded sparse terms to joint fusion without a term dimension."""
    if indices.shape[-1] > 6:
        return reference(features, indices, weights, left, right)
    inner = features.shape[-1] // (left * right)
    value = features.unflatten(-1, (left, inner, right))
    terms = [
        value.index_select(-2, indices[:, k].long()) * weights[..., None, :, k, None]
        for k in range(indices.shape[-1])
    ]
    return sum(terms[1:], terms[0]).flatten(-3)


def adjoint(features, indices, weights, gradient, left, right, needed):
    """Transpose the sparse map and reduce gradients over broadcast axes."""
    inner = features.shape[-1] // (left * right)
    g = gradient.unflatten(-1, (left, indices.shape[0], right))
    dx = dw = None
    if needed[0]:
        values = (g.unsqueeze(-2) * weights[..., None, :, :, None]).flatten(-3, -2)
        result = gradient.new_zeros((*g.shape[:-3], left, inner, right))
        dx = result.index_add(-2, indices.flatten().long(), values)
        dx = dx.flatten(-3).sum_to_size(features.shape)
    if needed[1]:
        x = features.unflatten(-1, (left, inner, right))[..., indices, :]
        dw = (g.unsqueeze(-2) * x).sum((-4, -1)).sum_to_size(weights.shape)
    return dx, dw


@torch.compiler.allow_in_graph
class SparseAxis(torch.autograd.Function):
    """Retain operands instead of the expanded sparse terms."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, indices, weights, left, right):
        return primal(features, indices, weights, left, right)

    @staticmethod
    def setup_context(ctx, inputs, output):
        features, indices, weights, ctx.left, ctx.right = inputs
        ctx.save_for_backward(features, indices, weights)
        ctx.save_for_forward(features, indices, weights)

    @staticmethod
    def backward(ctx, gradient):
        features, indices, weights = ctx.saved_tensors
        needed = torch_gradient_mask(
            ctx, (ctx.needs_input_grad[0], ctx.needs_input_grad[2])
        )
        dx, dw = adjoint(
            features, indices, weights, gradient, ctx.left, ctx.right, needed
        )
        return dx, None, dw, None, None

    @staticmethod
    def jvp(ctx, dx, unused_indices, dw, unused_left, unused_right):
        features, indices, weights = ctx.saved_tensors
        dx = torch.zeros_like(features) if dx is None else dx
        dw = torch.zeros_like(weights) if dw is None else dw
        return primal(dx, indices, weights, ctx.left, ctx.right) + primal(
            features, indices, dw, ctx.left, ctx.right
        )


@torch.compiler.allow_in_graph
def evaluate(features, indices, weights, left=1, right=1):
    """Expose the handwritten sparse transpose to AOTAutograd."""
    if nested_forward_ad():
        return primal(features, indices, weights, left, right)
    return SparseAxis.apply(features, indices, weights, left, right)
