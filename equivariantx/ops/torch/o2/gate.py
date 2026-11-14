"""Scalar activations and gate application in the declared storage layout."""

import torch

from ...common.torch.gate import expand_gates
from ....utils.layout import permute


def activate(module, features):
    """Evaluate the module's declared paths."""
    outputs = []
    for values, act in zip(features.split(module._sizes, dim=-1), module.acts):
        outputs.append(values if act is None else act(values))
    if outputs:
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
    return features[..., :0]


def evaluate(module, features):
    """Evaluate the module's declared paths."""
    inputs = features.split(module._input_sizes, dim=-1)
    scalars = module.act_scalars(module._select_entries(inputs, module._selections[0]))
    if not module._paths:
        return scalars
    gates = module.act_gates(module._select_entries(inputs, module._selections[1]))
    values = module._select_entries(inputs, module._selections[2])
    if module._broadcast_shape is not None:
        count, dim, mul = module._broadcast_shape
        output = (
            values.reshape(*features.shape[:-1], count, dim, mul)
            * gates.reshape(*features.shape[:-1], count, 1, mul)
        ).flatten(-3)
        return torch.cat((scalars, output), -1) if scalars.size(-1) else output
    if not module._gated_identity:
        values = permute(values, module._gated_indices, module._gated_inverse)
    if module._has_signs:
        values = values * module._gated_signs
    gates = expand_gates(
        gates,
        module._expanded_indices,
        module._output_indices,
        module._output_inverse,
        module._gate_indices,
        module._gate_inverse,
        module._gate_layout,
    )
    output = values * gates
    return torch.cat((scalars, output), dim=-1) if scalars.size(-1) else output
