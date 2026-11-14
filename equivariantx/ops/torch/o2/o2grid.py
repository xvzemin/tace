"""Circular transforms with sheet changes on the shorter coefficient axis."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.torch import grid


def coefficients(module, features, features2=None):
    if features2 is None:
        value = features.index_select(-1, module.input_index) * module.input_sign
        return value.unflatten(-1, module._coefficient_shape)
    shape = (len(module.time_parities), 2 * module.mmax + 1, module.num_channels)
    first = features.unflatten(-1, shape).movedim(-1, -3)
    second = features2.unflatten(-1, shape).movedim(-1, -3)
    second = second.index_select(-1, module.second_index) * module.second_sign
    return torch.stack((first, second), dim=-2).flatten(-3, -2)


def combine(module, value, split):
    value = value.unflatten(-2, (len(module.time_parities), 2))
    if split:
        first, second = value.unbind(-2)
        second = (second * module.second_sign).index_select(-1, module.second_index)
        return first.movedim(-3, -1).flatten(-3), second.movedim(-3, -1).flatten(-3)
    return value.flatten(-4).index_select(-1, module.output_index) * module.output_sign


def to_grid_reference(module, features, features2=None):
    value = coefficients(module, features, features2) @ module.synthesis.T
    value = value.unflatten(-2, (len(module.time_parities), 2))
    even, odd = value.unbind(-2)
    value = torch.stack((even + odd, even - odd), dim=-2) / math.sqrt(2)
    if len(module.time_parities) == 2:
        even, odd = value.unbind(-3)
        value = torch.stack((even + odd, even - odd), dim=-3) / math.sqrt(2)
    return value.flatten(-3, -2)


def from_grid_reference(module, features, split=False):
    value = features.unflatten(-2, (len(module.time_parities), 2))
    if len(module.time_parities) == 2:
        even, odd = value.unbind(-3)
        value = torch.stack((even + odd, even - odd), dim=-3) / math.sqrt(2)
    even, odd = value.unbind(-2)
    value = torch.stack((even + odd, even - odd), dim=-2) / math.sqrt(2)
    return combine(module, (value @ module.analysis.T).flatten(-3, -2), split)


def operands(module):
    return (
        module.input_index,
        module.output_index,
        module.input_sign,
        module.output_sign,
        module.num_sheets,
        2 * module.mmax + 1,
    )


def to_grid_program(module, features, features2, transform):
    if features2 is None:
        value = transform(features, *operands(module)).unflatten(
            -1, module._coefficient_shape
        )
    else:
        value = grid.mix(coefficients(module, features, features2))
    return value @ module.synthesis.T


def from_grid_program(module, features, split, transform):
    value = features @ module.analysis.T
    if split:
        return combine(module, grid.mix(value), True)
    return transform(value.flatten(-3), *operands(module), True)


def compiling():
    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def to_grid(module, features, features2=None):
    if config.TORCH_REFERENCE or not config.TORCH_O2_GRID or not compiling():
        return to_grid_reference(module, features, features2)
    return to_grid_program(module, features, features2, grid.evaluate)


def from_grid(module, features, split=False):
    if config.TORCH_REFERENCE or not config.TORCH_O2_GRID or not compiling():
        return from_grid_reference(module, features, split)
    return from_grid_program(module, features, split, grid.evaluate)
