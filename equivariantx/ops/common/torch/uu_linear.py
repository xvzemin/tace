"""Channel-preserving path sums with exact bilinear adjoints."""

import math

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ....utils.autograd import nested_forward_ad, torch_gradient_mask
from ....utils.metadata import parse_metadata
from .channel_linear import pack, unpack


def layout(storage, inputs, outputs, instructions, channels, mode):
    """Describe equivalent-copy paths independently of representation classes."""
    paths, offset = [], 0
    expanded = [0] * len(outputs)
    for instruction in instructions:
        i, j = instruction.i_in, instruction.i_out
        ni, no, _ = instruction.path_shape
        paths.append((i, j, offset, ni, no, instruction.path_weight, expanded[j]))
        offset += math.prod(instruction.path_shape)
        expanded[j] += ni * no * channels
    return repr((storage, tuple(inputs), tuple(outputs), channels, mode, tuple(paths)))


def primal(metadata, features, weight):
    """Contract short copy axes explicitly, leaving channels as vector operations."""
    storage, inputs, outputs, channels, mode, paths = parse_metadata(metadata)
    shape = torch.broadcast_shapes(features.shape[:-1], weight.shape[:-1])
    x = unpack(features, inputs, storage)
    parts = [[] for _ in outputs]
    for i, j, start, ni, no, scale, expanded in paths:
        v = x[i].unflatten(-1, (ni, channels))
        w = weight[..., start : start + ni * no * channels].reshape(
            *weight.shape[:-1], ni, no, channels
        )
        w = w * scale if scale != 1 else w
        if mode == "expand":
            value = (v.unsqueeze(-2) * w.unsqueeze(-4)).flatten(-3)
        elif ni <= 4:
            terms = [
                v[..., k, :].unsqueeze(-2) * w[..., k, :, :].unsqueeze(-3)
                for k in range(ni)
            ]
            value = sum(terms[1:], terms[0]).flatten(-2)
        else:
            value = torch.einsum("...diu,...iou->...dou", v, w).flatten(-2)
        parts[j].append(value)
    zero = features[..., :0].sum() + weight[..., :0].sum()
    values = []
    for (dim, mul), block in zip(outputs, parts):
        value = (
            (torch.cat(block, -1) if mode == "expand" else sum(block[1:], block[0]))
            if block
            else features.new_zeros((*shape, dim, mul)) + zero
        )
        values.append(value)
    return pack(values, storage) if values else features.new_zeros((*shape, 0)) + zero


def reduce_to(value, shape):
    """Sum known broadcast axes without asking whether a symbolic row count is one."""
    leading = value.ndim - len(shape)
    axes = tuple(range(leading)) + tuple(
        leading + i
        for i, size in enumerate(shape)
        if statically_known_true(size == 1)
        and not statically_known_true(value.shape[leading + i] == 1)
    )
    if axes:
        value = value.sum(axes, keepdim=True)
    return value.reshape(shape) if leading else value


def adjoint(metadata, features, weight, gradient, needed):
    """Reduce cotangents over copy, angular and broadcast dimensions exactly."""
    storage, inputs, outputs, channels, mode, paths = parse_metadata(metadata)
    x, g = unpack(features, inputs, storage), unpack(gradient, outputs, storage)
    dx, dw = [None] * len(inputs), []
    for i, j, start, ni, no, scale, expanded in paths:
        v = x[i].unflatten(-1, (ni, channels))
        w = weight[..., start : start + ni * no * channels].reshape(
            *weight.shape[:-1], ni, no, channels
        )
        if mode == "expand":
            grad = g[j][..., expanded : expanded + ni * no * channels].unflatten(
                -1, (ni, no, channels)
            )
            if needed[0]:
                value = (grad * w.unsqueeze(-4)).sum(-2) * scale
            if needed[1]:
                dw.append(((grad * v.unsqueeze(-2)).sum(-4) * scale).flatten(-3))
        else:
            grad = g[j].unflatten(-1, (no, channels))
            if needed[0]:
                if no <= 4:
                    terms = [
                        grad[..., k, :].unsqueeze(-2) * w[..., :, k, :].unsqueeze(-3)
                        for k in range(no)
                    ]
                    value = sum(terms[1:], terms[0]) * scale
                else:
                    value = torch.einsum("...dou,...iou->...diu", grad, w) * scale
            if needed[1]:
                if ni <= 4:
                    value_w = (
                        torch.stack(
                            [
                                (v[..., k, :].unsqueeze(-2) * grad).sum(-3)
                                for k in range(ni)
                            ],
                            -3,
                        )
                        * scale
                    )
                else:
                    value_w = torch.einsum("...diu,...dou->...iou", v, grad) * scale
                dw.append(value_w.flatten(-3))
        if needed[0]:
            value = value.flatten(-2)
            dx[i] = value if dx[i] is None else dx[i] + value
    zero = features[..., :0].sum() + weight[..., :0].sum() + gradient[..., :0].sum()
    if needed[0]:
        dx = [
            reduce_to(value, source.shape)
            if value is not None
            else torch.zeros_like(source) + zero
            for value, source in zip(dx, x)
        ]
        dx = pack(dx, storage) if dx else torch.zeros_like(features) + zero
    else:
        dx = None
    dw = (
        (
            reduce_to(torch.cat(dw, -1), weight.shape)
            if dw
            else torch.zeros_like(weight) + zero
        )
        if needed[1]
        else None
    )
    return dx, dw


@torch.compiler.allow_in_graph
class UuLinear(torch.autograd.Function):
    """Retain only original inputs and expose recursive Torch derivatives."""

    generate_vmap_rule = True

    @staticmethod
    def forward(metadata, features, weight):
        return primal(metadata, features, weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.kernel_metadata, features, weight = inputs
        ctx.save_for_backward(features, weight)
        ctx.save_for_forward(features, weight)

    @staticmethod
    def backward(ctx, gradient):
        features, weight = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[1:])
        return None, *adjoint(ctx.kernel_metadata, features, weight, gradient, needed)

    @staticmethod
    def jvp(ctx, unused, dx, dw):
        features, weight = ctx.saved_tensors
        dx = torch.zeros_like(features) if dx is None else dx
        dw = torch.zeros_like(weight) if dw is None else dw
        return primal(ctx.kernel_metadata, dx, weight) + primal(
            ctx.kernel_metadata, features, dw
        )


@torch.compiler.allow_in_graph
def evaluate(metadata, features, weight):
    """Make the handwritten derivative program available to AOTAutograd."""
    if nested_forward_ad():
        return primal(metadata, features, weight)
    return UuLinear.apply(metadata, features, weight)


def supported(features, weight):
    """Native row layout for shared or independently weighted node matrices."""
    return (
        features.ndim == 2
        and weight.ndim in (1, 2)
        and (
            weight.ndim == 1
            or statically_known_true(weight.shape[0] == 1)
            or statically_known_true(weight.shape[0] == features.shape[0])
        )
    )
