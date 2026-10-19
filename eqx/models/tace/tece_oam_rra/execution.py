"""Tiled rotary attention and target-node aggregation."""

from functools import lru_cache

from ....conv.attention import normalized_exp_forward
from ....conv.execution import TiledProgram, execution_plan
from ....conv.graph import reduce_messages

BACKWARD_TILE_SIZE = 4096


@lru_cache(maxsize=64)
def forward_plan(nodes, output):
    return execution_plan(repr((nodes, ((output, 0, "target"),))))


def forward(base, inputs, source, target, outputs, tile_size=8192):
    """Compute attention-weighted node updates in edge tiles."""
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
    scale = normalized_exp_forward(
        scores, inputs[4], target, result.shape[0], eps, denominator, maximum
    )
    scale.mul_(inputs[4].square())
    result.zero_()
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(value_plan, inputs, source, target, begin, end, shared)
        message = tile.value(value)
        reduce_messages(
            message, tile.target, result.flatten(1), scale[begin:end], channels
        )
        del message, tile
