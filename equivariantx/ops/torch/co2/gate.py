"""Scalar activations and gate application in the declared storage layout."""

import torch

from ....co2.tensor_product import quarter_turn


def activate(module, features):
    """Evaluate the module's declared paths."""
    values = [
        value if act is None else act(value)
        for value, act in zip(features.split(module.sizes, -1), module.activation.acts)
    ]
    return torch.cat(values, -1) if values else features[..., :0]


def evaluate(module, features):
    """Evaluate the module's declared paths."""
    inputs = features.split(module.sizes, -1)
    outputs = [
        inputs[i] if act is None else act(inputs[i])
        for i, act in zip(module._scalar_locations, module.act_scalars.acts)
    ]
    gates = [
        inputs[i] if act is None else act(inputs[i])
        for i, act in zip(module._gate_locations, module.act_gates.acts)
    ]
    for path in module._paths:
        ir = module.irreps_gated[path.i_gated]
        value = inputs[module._gated_locations[path.i_gated]].reshape(
            *features.shape[:-1], ir.mul, ir.ir.dim
        )
        value = value[..., path.gated_start : path.gated_start + path.mul, :]
        if path.ir_gate.is_odd_scalar() and path.ir_gated.m:
            value = quarter_turn(value)
        scalar = gates[path.i_gate][..., path.gate_start : path.gate_start + path.mul]
        outputs.append((value * scalar.unsqueeze(-1)).flatten(-2))
    return torch.cat(outputs, -1) if outputs else features[..., :0]
