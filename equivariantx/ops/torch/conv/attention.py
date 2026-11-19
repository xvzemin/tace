"""Segmented softmax and streaming attention in ordinary Torch operations."""

import torch


def reference(module, inputs):
    """Evaluate the weighted softmax and value sum on the complete graph."""
    result, denominator, maximum = module.fake_function(inputs)
    target = inputs[module.target_slot]
    target = target[1] if target.ndim == 2 else target
    scores, value = module.edge_function(inputs)
    scores = scores.masked_fill(~inputs[module.valid_slot][:, None], -torch.inf)
    maximum = maximum.fill_(-torch.inf).scatter_reduce(
        0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
    )
    maximum = torch.where(torch.isneginf(maximum), 0, maximum)
    weight = (scores - maximum[target]).exp() * inputs[module.normalizer_slot]
    denominator = denominator.zero_().index_add(0, target, weight) + module.eps
    nodes, angular, channels = result.shape
    heads = denominator.shape[1]
    weighted = (
        value.reshape(-1, angular, heads, channels // heads)
        * (weight * inputs[module.value_weight_slot])[:, None, :, None]
    )
    result = result.zero_().index_add(0, target, weighted.reshape_as(value))
    result = (
        result.reshape(nodes, angular, heads, channels // heads)
        / denominator[:, None, :, None]
    )
    return result.flatten(2), denominator, maximum


def stream(module, inputs, create_graph):
    """Accumulate an online softmax over bounded edge tiles."""
    result, denominator, maximum = module.fake_function(inputs)
    result.zero_()
    denominator.zero_()
    maximum.fill_(-torch.inf)
    nodes, angular, channels = result.shape
    heads = denominator.shape[1]
    for tile, values in enumerate(module.tiles(inputs)):
        index = values[module.target_slot]
        target = index[1] if index.ndim == 2 else index
        scores, value = module.edge_function(values)
        scores = torch.where(values[module.valid_slot][:, None], scores, -torch.inf)
        next_max = maximum.scatter_reduce(
            0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
        )
        shift = torch.where(torch.isneginf(next_max), 0, next_max)
        if tile:
            scale = (maximum - shift).exp()
            result = (
                result.view(nodes, angular, heads, channels // heads)
                * scale[:, None, :, None]
            ).reshape_as(result)
            denominator = denominator * scale
        exp = (scores - shift[target]).exp()
        weight = exp * values[module.normalizer_slot]
        weighted_value = value.view(-1, angular, heads, channels // heads)
        weighted_value = (
            weighted_value
            * (weight * values[module.value_weight_slot])[:, None, :, None]
        )
        result = result.index_add(0, target, weighted_value.reshape_as(value))
        denominator = denominator.index_add(0, target, weight)
        maximum = next_max
    denominator = denominator + module.eps
    result = (
        result.view(nodes, angular, heads, channels // heads)
        / denominator[:, None, :, None]
    ).reshape_as(result)
    return result, denominator, shift
