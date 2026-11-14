"""Exact sheet changes and constant grid projections."""

import math

import torch

from ....utils.autograd import forward_ad_active


def mix(features):
    """Apply the normalized Walsh transform to two or four sheets."""
    sheets = features.shape[-2]
    if sheets == 4:
        value = features.unflatten(-2, (2, 2))
        even, odd = value.unbind(-3)
        features = (torch.stack((even + odd, even - odd), -3) / math.sqrt(2)).flatten(
            -3, -2
        )
    value = features.unflatten(-2, (sheets // 2, 2))
    even, odd = value.unbind(-2)
    return (torch.stack((even + odd, even - odd), -2) / math.sqrt(2)).flatten(-3, -2)


def primal(
    features, input_index, output_index, input_sign, output_sign, sheets, width, reverse
):
    """Fuse a signed feature permutation with the sheet basis change."""
    value = features if reverse else features.index_select(-1, input_index) * input_sign
    value = mix(
        value.unflatten(-1, (features.shape[-1] // (sheets * width), sheets, width))
    ).flatten(-3)
    return value.index_select(-1, output_index) * output_sign if reverse else value


@torch.compiler.allow_in_graph
class SheetTransform(torch.autograd.Function):
    """Save only the fixed permutation; transpose the complete linear program."""

    generate_vmap_rule = True

    @staticmethod
    def forward(
        features,
        input_index,
        output_index,
        input_sign,
        output_sign,
        sheets,
        width,
        reverse,
    ):
        return primal(
            features,
            input_index,
            output_index,
            input_sign,
            output_sign,
            sheets,
            width,
            reverse,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, *tables, ctx.sheets, ctx.width, ctx.reverse = inputs
        ctx.save_for_backward(*tables)
        ctx.save_for_forward(*tables)
        ctx.shape, ctx.dtype, ctx.device = output.shape, output.dtype, output.device

    @staticmethod
    def backward(ctx, gradient):
        value = evaluate(
            gradient, *ctx.saved_tensors, ctx.sheets, ctx.width, not ctx.reverse
        )
        return value, None, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, gradient, *unused):
        if gradient is None:
            return torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        return primal(gradient, *ctx.saved_tensors, ctx.sheets, ctx.width, ctx.reverse)


@torch.compiler.allow_in_graph
def evaluate(
    features,
    input_index,
    output_index,
    input_sign,
    output_sign,
    sheets,
    width,
    reverse=False,
):
    """Use an exact transpose at every reverse derivative order."""
    if forward_ad_active():
        return primal(
            features,
            input_index,
            output_index,
            input_sign,
            output_sign,
            sheets,
            width,
            reverse,
        )
    return SheetTransform.apply(
        features,
        input_index,
        output_index,
        input_sign,
        output_sign,
        sheets,
        width,
        reverse,
    )


def linear_reference(features, matrix):
    """Apply the complete grid matrix with ordinary Torch differentiation."""
    return features @ matrix.T


@torch.compiler.allow_in_graph
class GridLinear(torch.autograd.Function):
    """Keep only a fixed basis matrix for every reverse derivative order."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, matrix):
        return linear_reference(features, matrix)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, matrix = inputs
        ctx.save_for_backward(matrix)
        ctx.save_for_forward(matrix)
        ctx.shape, ctx.dtype, ctx.device = output.shape, output.dtype, output.device

    @staticmethod
    def backward(ctx, gradient):
        (matrix,) = ctx.saved_tensors
        return linear(gradient, matrix.T), None

    @staticmethod
    def jvp(ctx, gradient, unused_matrix):
        if gradient is None:
            return torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        return linear_reference(gradient, ctx.saved_tensors[0])


@torch.compiler.allow_in_graph
def linear(features, matrix):
    if matrix.requires_grad or forward_ad_active():
        return linear_reference(features, matrix)
    return GridLinear.apply(features, matrix)
