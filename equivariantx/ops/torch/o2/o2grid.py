"""Circular transforms with sheet changes on the shorter coefficient axis."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
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


def operands(module, split=False):
    return (
        module.split_input_index if split else module.input_index,
        module.split_output_index if split else module.output_index,
        module.input_sign,
        module.split_output_sign if split else module.output_sign,
        module.num_sheets,
        2 * module.mmax + 1,
    )


def to_grid_program(module, features, features2, transform):
    split = features2 is not None
    if split:
        features = torch.cat((features, features2), -1)
    value = transform(features, *operands(module, split)).unflatten(
        -1, module._coefficient_shape
    )
    return value @ module.synthesis.T


def from_grid_program(module, features, split, transform):
    value = features @ module.analysis.T
    value = transform(value.flatten(-3), *operands(module, split), True)
    return value.chunk(2, dim=-1) if split else value


def compiling():
    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def band_map(matrix, values):
    """Apply a shared band matrix without expanding it over channels or batch."""
    flat = values.reshape(-1, *values.shape[-2:])
    out = torch.bmm(matrix.expand(flat.shape[0], -1, -1), flat)
    return out.reshape(*values.shape[:-2], matrix.shape[0], values.shape[-1])


def band_input(module, features, split=False):
    index = module.split_in_bands if split else module.in_bands
    if not split and not module.band_layout and index is not None:
        features = features.index_select(-1, index)
    values = features.unflatten(-1, (-1, module.num_channels))
    if (split or module.band_layout) and index is not None:
        values = values.index_select(-2, index)
    return values


def band_output(module, values, split=False):
    index = module.split_out_bands if split else module.out_bands
    if (split or module.band_layout) and index is not None:
        values = values.index_select(-2, index)
    values = values.flatten(-2)
    if not split and not module.band_layout and index is not None:
        values = values.index_select(-1, index)
    return values.chunk(2, dim=-1) if split else values


def band_synthesis(module, features, features2=None, transform=band_map):
    split = features2 is not None
    if split:
        features = torch.cat((features, features2), -1)
    values = transform(module.synthesis_bands, band_input(module, features, split))
    return values.transpose(-1, -2).unflatten(
        -1, (module.num_sheets, module.resolution)
    )


def band_analysis(module, features, split=False, transform=band_map):
    coeff = transform(module.analysis_bands.T, features.flatten(-2).transpose(-1, -2))
    return band_output(module, coeff, split)


def use_bands(module):
    return config.O2_GRID_FUSED and module.dense_grid


def to_grid(module, features, features2=None):
    if config.TORCH_REFERENCE or not config.TORCH_O2_GRID:
        return to_grid_reference(module, features, features2)
    if use_bands(module):
        return band_synthesis(module, features, features2)
    if not compiling():
        return to_grid_reference(module, features, features2)
    return to_grid_program(module, features, features2, grid.evaluate)


def from_grid(module, features, split=False):
    if config.TORCH_REFERENCE or not config.TORCH_O2_GRID:
        return from_grid_reference(module, features, split)
    if use_bands(module):
        return band_analysis(module, features, split)
    if not compiling():
        return from_grid_reference(module, features, split)
    return from_grid_program(module, features, split, grid.evaluate)


def sample_activation(values, act):
    out = act(values)
    if out.shape != values.shape:
        raise ValueError("A grid activation must preserve the sampled tensor shape.")
    return out


def activate(module, x, act):
    if act is torch.square and config.TORCH_O2_GRID and not config.TORCH_REFERENCE:
        return product(module, x, x)
    return from_grid(module, sample_activation(to_grid(module, x), act))


class BandProduct(torch.autograd.Function):
    """Multiply on the grid without saving sampled tensors for the adjoint."""

    @staticmethod
    def forward(ctx, a, b, out, x, y):
        ctx.save_for_backward(a, b, out, x, y)
        width, channels = x.shape[-2:]
        xs, ys = x.reshape(-1, width, channels), y.reshape(-1, width, channels)
        tile = max(
            1,
            config.O2_GRID_TILE_BYTES // (2 * a.shape[0] * channels * x.element_size()),
        )

        def evaluate(xs, ys, destination=None):
            u = band_map(a, xs)
            if x is y and a is b:
                u.square_()
            else:
                u.mul_(band_map(b, ys))
            if destination is None:
                return band_map(out.T, u)
            return torch.bmm(out.T.expand(xs.shape[0], -1, -1), u, out=destination)

        if xs.shape[0] <= tile:
            return evaluate(xs, ys).reshape(x.shape)
        result = torch.empty_like(xs, memory_format=torch.contiguous_format)
        for lo in range(0, xs.shape[0], tile):
            evaluate(xs[lo : lo + tile], ys[lo : lo + tile], result[lo : lo + tile])
        return result.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad):
        a, b, out, x, y = ctx.saved_tensors
        dx = BandProduct.apply(out, b, a, grad, y) if ctx.needs_input_grad[3] else None
        if ctx.needs_input_grad[4]:
            dy = (
                dx
                if dx is not None and x is y and a is b
                else BandProduct.apply(out, a, b, grad, x)
            )
        else:
            dy = None
        return None, None, None, dx, dy


def product(module, x, y):
    if (
        config.TORCH_O2_GRID
        and not config.TORCH_REFERENCE
        and use_bands(module)
        and x.shape == y.shape
        and not compiling()
        and not forward_ad_active()
        and (
            config.TORCH_RECOMPUTE
            or 2 * x.numel() * x.element_size() * module.resolution
            > config.O2_GRID_TILE_BYTES * (2 * module.mmax + 1)
        )
    ):
        a = module.synthesis_bands
        u = band_input(module, x)
        v = u if x is y else band_input(module, y)
        return band_output(module, BandProduct.apply(a, a, module.analysis_bands, u, v))
    if x is y and not config.TORCH_REFERENCE and config.TORCH_O2_GRID:
        return from_grid(module, to_grid(module, x).square())
    return from_grid(module, to_grid(module, x) * to_grid(module, y))
