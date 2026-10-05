"""Receiver-normalized rotary attention with bounded edge tiles."""

from functools import lru_cache

import torch

from ....conv.execution import TiledProgram, execution_plan

BACKWARD_TILE_SIZE = 4096


@lru_cache(maxsize=64)
def forward_plan(nodes, output):
    return execution_plan(repr((nodes, ((output, 0, "target"),))))


def forward(base, inputs, source, target, outputs, tile_size=8192):
    """Retain only edge scores between the score and message passes."""
    nodes, score, value, _, heads, channels, eps = base
    score_plan = forward_plan(nodes, score)
    value_plan = forward_plan(nodes, value)
    result, denominator, maximum = outputs
    shared = {}
    scores = inputs[0].new_empty((source.numel(), heads))
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(score_plan, inputs, source, target, begin, end, shared)
        scores[begin:end] = tile.value(score)
        del tile
    maximum.fill_(-torch.inf)
    maximum.scatter_reduce_(0, target[:, None].expand_as(scores), scores, reduce="amax")
    maximum.masked_fill_(maximum.isneginf(), 0)
    exponential = (scores - maximum[target]).exp()
    denominator.zero_().index_add_(0, target, exponential * inputs[4]).add_(eps)
    scale = exponential * inputs[4].square() / denominator[target]
    result.zero_()
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(value_plan, inputs, source, target, begin, end, shared)
        message = tile.value(value).reshape(end - begin, -1, heads, channels // heads)
        message = (message * scale[begin:end, None, :, None]).flatten(1)
        result.flatten(1).index_add_(0, tile.target, message)
        del message, tile
