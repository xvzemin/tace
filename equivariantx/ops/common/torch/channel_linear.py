"""Shared channel maps with operand-only storage and exact bilinear adjoints."""

import torch

from ....utils.autograd import nested_forward_ad, torch_gradient_mask
from ....utils.metadata import parse_metadata
from .linear import linear, linear_reference


def unpack(features, shapes, storage):
    """View flattened features as component-major channel blocks."""
    values = features.split([dim * mul for dim, mul in shapes], -1)
    return [
        value.reshape(*features.shape[:-1], dim, mul)
        if storage == "ir_mul"
        else value.reshape(*features.shape[:-1], mul, dim).transpose(-1, -2)
        for value, (dim, mul) in zip(values, shapes)
    ]


def pack(values, storage):
    """Restore the original component/channel storage convention."""
    values = [
        value if storage == "ir_mul" else value.transpose(-1, -2) for value in values
    ]
    return torch.cat([value.flatten(-2) for value in values], -1)


def primal(metadata, features, weight, bias, *, ordinary=False):
    """Use grouped GEMMs or the independent path-wise Torch reference."""
    storage, layout = parse_metadata(metadata)
    inputs = unpack(features, layout[0], storage)
    function = linear_reference if ordinary else linear
    outputs = function(inputs, weight, bias, layout, features.shape[:-1])
    return pack(outputs, storage)


def adjoint(metadata, features, weight, gradient, needed, bias_numel):
    """Differentiate each independent channel map without retaining packed inputs."""
    storage, (input_shapes, output_shapes, paths, biases, groups) = parse_metadata(
        metadata
    )
    inputs = unpack(features, input_shapes, storage)
    outputs = unpack(gradient, output_shapes, storage)
    zero = features[..., :0].sum() + weight[..., :0].sum() + gradient[..., :0].sum()
    dx = [None] * len(input_shapes)
    dw = []
    for i, j, start, scale in paths:
        ci, co = input_shapes[i][1], output_shapes[j][1]
        if needed[0]:
            matrix = weight[start : start + ci * co].reshape(ci, co)
            matrix = matrix * scale if scale != 1 else matrix
            value = outputs[j] @ matrix.T
            dx[i] = value if dx[i] is None else dx[i] + value
        if needed[1]:
            left = inputs[i].reshape(-1, ci)
            right = outputs[j].reshape(-1, co)
            dw.append(((left.T @ right) * scale).flatten())
    if needed[0]:
        dx = pack(
            [
                value if value is not None else torch.zeros_like(source) + zero
                for value, source in zip(dx, inputs)
            ],
            storage,
        )
    else:
        dx = None
    dw = (
        torch.cat(dw)
        if needed[1] and dw
        else torch.zeros_like(weight) + zero
        if needed[1]
        else None
    )
    if needed[2]:
        db = [outputs[j].reshape(-1, size).sum(0) for j, start, size in biases]
        db = torch.cat(db) if db else features.new_zeros((bias_numel,)) + zero
    else:
        db = None
    return dx, dw, db


@torch.compiler.allow_in_graph
class ChannelLinear(torch.autograd.Function):
    """Save original operands, keeping higher-order derivatives in Torch algebra."""

    generate_vmap_rule = True

    @staticmethod
    def forward(metadata, features, weight, bias):
        return primal(metadata, features, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.kernel_metadata, features, weight, bias = inputs
        ctx.bias_numel = bias.numel()
        ctx.save_for_backward(features, weight)
        ctx.save_for_forward(features, weight)

    @staticmethod
    def backward(ctx, gradient):
        features, weight = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[1:])
        return None, *adjoint(
            ctx.kernel_metadata, features, weight, gradient, needed, ctx.bias_numel
        )

    @staticmethod
    def jvp(ctx, unused, dx, dw, db):
        features, weight = ctx.saved_tensors
        dx = torch.zeros_like(features) if dx is None else dx
        dw = torch.zeros_like(weight) if dw is None else dw
        bias = features.new_zeros((ctx.bias_numel,))
        db = bias if db is None else db
        return primal(ctx.kernel_metadata, dx, weight, bias) + primal(
            ctx.kernel_metadata, features, dw, db
        )


@torch.compiler.allow_in_graph
def evaluate(metadata, features, weight, bias):
    """Expose the complete adjoint to AOTAutograd and preserve nested JVPs."""
    if nested_forward_ad():
        return primal(metadata, features, weight, bias)
    return ChannelLinear.apply(metadata, features, weight, bias)


def supported(features, weight, bias, layout):
    """Select flat shared-parameter maps with nonempty channel paths."""
    return (
        features.ndim == 2
        and weight.ndim == 1
        and bias.ndim == 1
        and bool(layout[2])
        and bool(layout[1])
        and all(dim > 0 and mul > 0 for dim, mul in (*layout[0], *layout[1]))
    )
