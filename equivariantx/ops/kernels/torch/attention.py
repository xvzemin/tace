"""Segmented normalization with ordinary, higher-order Torch derivatives."""

import torch


def normalized_exp(scores, weight, target, num_nodes, eps):
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
