"""Segmented normalization with compact reverse derivatives."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config


def normalized_exp(scores, weight, target, num_nodes, eps):
    """Retain compact eager adjoints and expose ordinary operations to compilers."""
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_ATTENTION_ADJOINT
        or torch.compiler.is_compiling()
        or is_fake(scores)
        or get_proxy_mode() is not None
        or forward_ad_active()
        or not torch.is_grad_enabled()
        or not (scores.requires_grad or weight.requires_grad)
        or (
            not scores.is_cuda
            and scores.numel() < config.TORCH_ATTENTION_MIN_CPU_ELEMENTS
        )
    ):
        return normalized_exp_reference(scores, weight, target, num_nodes, eps)
    return NormalizedExp.apply(scores, weight, target, num_nodes, eps)


class NormalizedExp(torch.autograd.Function):
    """Save normalized values for recursively differentiable reverse passes."""

    @staticmethod
    def forward(ctx, scores, weight, target, num_nodes, eps):
        result = normalized_exp_reference(scores, weight, target, num_nodes, eps)
        ctx.num_nodes = num_nodes
        ctx.save_for_backward(result, weight, target)
        return result

    @staticmethod
    def backward(ctx, grad):
        values, weight, target = ctx.saved_tensors
        total = values.new_zeros((ctx.num_nodes, values.shape[1])).index_add(
            0, target, values * grad
        )[target]
        scores_grad = (
            values * (grad - weight * total) if ctx.needs_input_grad[0] else None
        )
        weight_grad = (
            (-values * total).sum_to_size(weight.shape)
            if ctx.needs_input_grad[1]
            else None
        )
        return scores_grad, weight_grad, None, None, None


def normalized_exp_reference(scores, weight, target, num_nodes, eps):
    """Return shifted exponentials divided by their weighted node sums."""
    maxima = scores.new_full((num_nodes, scores.shape[1]), -torch.inf)
    maxima = maxima.scatter_reduce(
        0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
    )
    maxima = torch.where(torch.isneginf(maxima), 0, maxima)
    exponential = (scores - maxima[target]).exp()
    denominator = scores.new_zeros((num_nodes, scores.shape[1])).index_add(
        0, target, exponential * weight
    )
    return exponential / (denominator[target] + eps)


def evaluate_adjoint(values, grad, weight, target, num_nodes):
    """Return score, weight and node-sum terms for the native transpose ABI."""
    total = values.new_zeros((num_nodes, values.shape[1])).index_add(
        0, target, values * grad
    )
    selected = total[target]
    return (
        values * (grad - weight * selected),
        (-values * selected).sum_to_size(weight.shape),
        total,
    )
