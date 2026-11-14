"""Segmented softmax and streaming attention in ordinary Torch operations."""

import torch

from ... import get_kernel


def graph_softmax(scores, target, num_nodes, weight, eps):
    """Evaluate the shared Torch normalization and its analytic adjoint."""
    return (
        get_kernel("graph_normalized_exp", "torch")(
            scores, weight, target, num_nodes, eps
        )
        * weight
    )


def stream(module, inputs, create_graph):
    """Accumulate an online softmax over bounded edge tiles."""
    result, denominator, maximum = module.fake_function(inputs)
    result.zero_()
    denominator.zero_()
    maximum.fill_(-torch.inf)
    nodes, angular, channels = result.shape
    heads = denominator.shape[1]
    for values in module.tiles(inputs):
        index = values[module.target_slot]
        target = index[1] if index.ndim == 2 else index
        scores, value = module.edge_function(values)
        scores = torch.where(values[module.valid_slot][:, None], scores, -torch.inf)
        local_max = torch.full_like(maximum, -torch.inf).scatter_reduce(
            0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
        )
        next_max = torch.maximum(maximum, local_max)
        shift = torch.where(torch.isneginf(next_max), 0, next_max)
        scale = (maximum - shift).exp()
        exp = (scores - shift[target]).exp()
        weight = exp * values[module.normalizer_slot]
        weighted_value = value.view(-1, angular, heads, channels // heads)
        weighted_value = (
            weighted_value
            * (weight * values[module.value_weight_slot])[:, None, :, None]
        )
        result = (
            result.view(nodes, angular, heads, channels // heads)
            * scale[:, None, :, None]
        ).reshape_as(result)
        result = result.index_add(0, target, weighted_value.reshape_as(value))
        denominator = (denominator * scale).index_add(0, target, weight)
        maximum = next_max
    denominator = denominator + module.eps
    result = (
        result.view(nodes, angular, heads, channels // heads)
        / denominator[:, None, :, None]
    ).reshape_as(result)
    return (result, denominator, torch.where(torch.isneginf(maximum), 0, maximum))
