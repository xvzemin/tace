"""Receiver-normalized rotary attention with bounded edge tiles."""

from functools import lru_cache

import torch

from ....conv.execution import TiledProgram, execution_plan


@lru_cache(maxsize=64)
def forward_plan(base):
    nodes, score, value, *_ = base
    return execution_plan(repr((nodes, ((score, 0, "target"), (value, 1, "target")))))


def forward(base, inputs, source, target, outputs, tile_size=4096):
    """Retain only edge scores between the score and message passes."""
    nodes, score, value, _, heads, channels, eps = base
    plan = forward_plan(base)
    result, denominator, maximum = outputs
    shared = {}
    scores = inputs[0].new_empty((source.numel(), heads))
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(plan, inputs, source, target, begin, end, shared)
        scores[begin:end] = tile.value(score)
    maximum.fill_(-torch.inf)
    maximum.scatter_reduce_(0, target[:, None].expand_as(scores), scores, reduce="amax")
    maximum.masked_fill_(maximum.isneginf(), 0)
    exponential = (scores - maximum[target]).exp()
    denominator.zero_().index_add_(0, target, exponential * inputs[4]).add_(eps)
    scale = exponential * inputs[4].square() / denominator[target]
    result.zero_()
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(plan, inputs, source, target, begin, end, shared)
        message = tile.value(value).reshape(end - begin, -1, heads, channels // heads)
        message = (message * scale[begin:end, None, :, None]).flatten(1)
        result.flatten(1).index_add_(0, tile.target, message)
