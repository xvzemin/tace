"""Channel maps in the module's representation layout."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.torch import channel_linear
from ...common.torch import uu_linear as channelwise
from ...common.torch.linear import linear as contract


def linear(module, features, weight, bias):
    """Evaluate the declared Linear paths and output layout."""
    if (
        not config.TORCH_REFERENCE
        and config.TORCH_CHANNEL_LINEAR
        and hasattr(module, "channel_metadata")
        and channel_linear.supported(features, weight, bias, module.layout)
    ):
        output = channel_linear.evaluate(
            module.channel_metadata, features, weight, bias
        )
        return module.projection(output)

    shape = torch.broadcast_shapes(
        features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
    )
    inputs = [
        features[..., section].reshape(*features.shape[:-1], mul, dim).transpose(-1, -2)
        for mul, dim, section in module.input_paths
    ]
    outputs = contract(inputs, weight, bias, module.layout, shape)
    output = (
        torch.cat([v.transpose(-1, -2).flatten(-2) for v in outputs], -1)
        if outputs
        else features.new_zeros((*shape, 0))
        + features[..., :0].sum()
        + weight[..., :0].sum()
        + bias[..., :0].sum()
    )
    return module.projection(output)


def uu_linear(module, features, weight):
    """Evaluate the declared UuLinear paths and output layout."""
    if (
        not config.TORCH_REFERENCE
        and config.TORCH_UU_LINEAR
        and hasattr(module, "uu_metadata")
    ):
        if config.TORCH_UU_ADJOINT:
            return channelwise.evaluate(module.uu_metadata, features, weight)
        if module.path_mode == "sum" and (
            features.is_cuda
            or torch.compiler.is_compiling()
            or get_proxy_mode() is not None
        ):
            return channelwise.primal(module.uu_metadata, features, weight)
    return uu_reference(module, features, weight)


def uu_reference(module, features, weight):
    """Independent ordinary einsum implementation of channelwise paths."""
    shape = torch.broadcast_shapes(features.shape[:-1], weight.shape[:-1])
    outputs = [[] for _ in module.irreps_out]
    offset = 0
    for ins in module.instructions:
        _, ir = module.irreps_in[ins.i_in]
        ni, no, channels = ins.path_shape
        x = features[..., module.slices[ins.i_in]].reshape(
            *features.shape[:-1], ni, channels, ir.dim
        )
        size = ni * no * channels
        w = weight[..., offset : offset + size].reshape(
            *weight.shape[:-1], ni, no, channels
        )
        offset += size
        value = (
            torch.einsum(
                "...iud,...iou->...ioud"
                if module.path_mode == "expand"
                else "...iud,...iou->...oud",
                x,
                w,
            )
            * ins.path_weight
        )
        outputs[ins.i_out].append(
            value.flatten(-4, -2)
            if module.path_mode == "expand"
            else value.flatten(-3, -2)
        )
    zero = features[..., :0].sum() + weight[..., :0].sum()
    values = []
    for (mul, ir), parts in zip(module.irreps_out, outputs):
        value = (
            (
                torch.cat(parts, -2)
                if module.path_mode == "expand"
                else sum(parts[1:], parts[0])
            )
            if parts
            else features.new_zeros((*shape, mul, ir.dim)) + zero
        )
        values.append(value.flatten(-2))
    return torch.cat(values, -1) if values else features.new_zeros((*shape, 0)) + zero
