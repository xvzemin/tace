"""Channel maps in the module's representation layout."""

import torch

from ... import config
from ...common.torch import channel_linear
from ...common.torch import uu_linear as channelwise
from ...common.torch.linear import linear as contract


def linear(module, features, weight, bias, leading_shape):
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
        return output

    inputs = [
        value.reshape(*features.shape[:-1], ir.dim, mul)
        for (ir, mul), value in zip(
            module.irreps_in, features.split(module._input_sizes, dim=-1)
        )
    ]
    outputs = [
        value.flatten(-2)
        for value in contract(inputs, weight, bias, module.layout, leading_shape)
    ]
    if not inputs:
        outputs = [value + features[..., :0].sum() for value in outputs]
    if outputs:
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
    return features.new_empty((*leading_shape, 0)) + (
        features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
    )


def uu_linear(module, features, weight):
    """Evaluate the declared UuLinear paths and output layout."""
    if (
        not config.TORCH_REFERENCE
        and config.TORCH_UU_LINEAR
        and hasattr(module, "uu_metadata")
    ):
        if config.TORCH_UU_ADJOINT:
            return channelwise.evaluate(module.uu_metadata, features, weight)
        if module.path_mode == "sum":
            return channelwise.primal(module.uu_metadata, features, weight)
    return uu_reference(module, features, weight)


def uu_reference(module, features, weight):
    """Independent ordinary einsum implementation of channelwise paths."""
    leading_shape = torch.broadcast_shapes(features.shape[:-1], weight.shape[:-1])
    inputs = [
        values.reshape(
            *features.shape[:-1], ir.dim, mul // module.num_channel, module.num_channel
        )
        for (ir, mul), values in zip(
            module.irreps_in, features.split(module._input_sizes, dim=-1)
        )
    ]
    weights = weight.split(module._weight_sizes, dim=-1)
    outputs = []
    zero = None
    for i_out, (ir, mul) in enumerate(module.irreps_out):
        contributions = []
        for index in module._instructions_by_output[i_out]:
            instruction = module.instructions[index]
            matrix = weights[index].reshape(*weight.shape[:-1], *instruction.path_shape)
            contributions.append(
                torch.einsum(
                    "...diu,...iou->...diou"
                    if module.path_mode == "expand"
                    else "...diu,...iou->...dou",
                    inputs[instruction.i_in],
                    matrix,
                )
                * instruction.path_weight
            )
        if contributions:
            output = (
                torch.cat([x.flatten(-3) for x in contributions], dim=-1)
                if module.path_mode == "expand"
                else sum(contributions[1:], contributions[0])
            )
            outputs.append(output.reshape(*leading_shape, ir.dim * mul))
        else:
            if zero is None:
                zero = features[..., :0].sum() + weight[..., :0].sum()
            outputs.append(features.new_zeros((*leading_shape, ir.dim * mul)) + zero)
    if outputs:
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
    return features.new_empty((*leading_shape, 0)) + (
        features[..., :0].sum() + weight[..., :0].sum()
    )
