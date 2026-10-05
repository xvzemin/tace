"""Radial networks evaluated inside tiled convolution programs."""

import math

import torch

from ..utils.metadata import parse_metadata
from .program import Program, activate


def radial_program(network, inputs, program, value):
    """Append a radial network to an edge expression graph.

    Parameters
    ----------
    network : torch.nn.Module
        Sequential scalar network. Linear layers, smooth activations, layer
        normalization and RMS normalization are supported.
    inputs : list of torch.Tensor
        Program inputs. Network parameters are appended without copying them.
    program : Program
        Destination expression graph.
    value : int
        Input expression index.
    """

    def parameter(tensor):
        slot = len(inputs)
        inputs.append(tensor.reshape(1, -1))
        return program.input(slot, "shared", int(tensor.numel()))

    layers = network.mlp if hasattr(network, "mlp") else network
    for layer in layers:
        if isinstance(layer, torch.nn.Sequential):
            value = radial_program(layer, inputs, program, value)
        elif isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            width = program.size(value)
            if tuple(layer.normalized_shape) != (width,):
                raise ValueError("Radial normalization must act on the channel axis.")
            eps = layer.eps
            if eps is None:
                eps = torch.finfo(inputs[0].dtype).eps
            value = program.add(
                "normalize",
                width,
                (value,),
                (eps, isinstance(layer, torch.nn.LayerNorm)),
            )
            if layer.weight is not None:
                value = program.binary("mul", value, parameter(layer.weight))
            if getattr(layer, "bias", None) is not None:
                value = program.binary("add", value, parameter(layer.bias))
        elif hasattr(layer, "weight") or hasattr(layer, "weights"):
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.T
            elif hasattr(layer, "get_weight"):
                weight = layer.get_weight()
            elif hasattr(layer, "weights"):
                weight = layer.weights
            else:
                weight = layer.weight
                if hasattr(layer, "alpha"):
                    weight = weight * layer.alpha
            if hasattr(layer, "h_in"):
                scale = layer.h_in * layer.var_in
                if layer.act is None:
                    scale /= layer.var_out
                weight = weight / math.sqrt(scale)
            if weight.ndim != 2 or weight.shape[0] != program.size(value):
                raise ValueError("Radial linear weight has an incompatible shape.")
            value = program.matmul(
                value, parameter(weight), 1, *(int(dim) for dim in weight.shape)
            )
            if getattr(layer, "bias", None) is not None:
                value = program.binary("add", value, parameter(layer.bias))
            if hasattr(layer, "h_in") and layer.act is not None:
                value = program.scale(
                    activate(program, value, layer.act), math.sqrt(layer.var_out)
                )
        else:
            value = activate(program, value, layer)
    return value


def compose_radial(metadata, inputs, slot, network):
    """Replace an edge input by a tiled radial network evaluation."""
    nodes, outputs = parse_metadata(metadata)
    inputs = list(inputs)
    program = Program()
    radial = inputs[slot]
    if isinstance(radial, tuple):
        parts = []
        for i, (feature, storage) in enumerate(radial):
            index = slot if i == 0 else len(inputs)
            if i == 0:
                inputs[slot] = feature
            else:
                inputs.append(feature)
            parts.append(program.input(index, storage, int(feature.shape[-1])))
        value = program.concatenate(parts)
    else:
        value = program.input(slot, "edge", int(radial.shape[-1]))
    value = radial_program(network, inputs, program, value)
    translated = {}
    for i, (op, size, args, data) in enumerate(nodes):
        if op == "input" and data[0] == slot:
            if size != program.size(value):
                raise ValueError(
                    "Radial network output does not match convolution weights."
                )
            translated[i] = value
        else:
            translated[i] = program.add(
                op, size, tuple(translated[j] for j in args), data
            )
    outputs = tuple((translated[root], output, kind) for root, output, kind in outputs)
    return repr((tuple(program.nodes), outputs)), inputs


def convolve(kind, metadata, operands, edge_index, network):
    """Evaluate a full radial network and tensor product in bounded edge tiles."""
    from .edge import evaluate

    output_role = 4 if kind == "o3" else 6
    output = operands[output_role]
    inputs = []
    program = Program()
    values = []
    for role, tensor in enumerate(operands):
        if role == 1 and isinstance(tensor, tuple):
            parts = []
            for feature, storage in tensor:
                if torch.compiler.is_compiling():
                    torch._dynamo.mark_static(feature, -1)
                parts.append(program.input(len(inputs), storage, int(feature.size(-1))))
                inputs.append(feature)
            values.append(program.concatenate(parts))
            continue
        if torch.compiler.is_compiling():
            torch._dynamo.mark_static(tensor, -1)
            if role == 2:
                torch._dynamo.mark_static(tensor, 0)
        if role == output_role:
            values.append(program.constant(int(tensor.size(-1)), 0))
            continue
        storage = (
            "source"
            if role == 0
            else "shared"
            if role == 2 or tensor.shape[0] == 1
            else "edge"
        )
        # Program widths are model constants, unlike dynamic node/edge counts.
        width = int(tensor.numel() if storage == "shared" else tensor.size(-1))
        values.append(program.input(len(inputs), storage, width))
        inputs.append(tensor.reshape(1, -1) if storage == "shared" else tensor)
    values[1] = radial_program(network, inputs, program, values[1])
    width = operands[2].shape[0] or operands[2].shape[1]
    if program.size(values[1]) + 1 == width:
        values[1] = program.concatenate((values[1], program.constant(1, 1)))
    if program.size(values[1]) != width:
        raise ValueError("Radial network output does not match projection input.")
    root = program.add(
        "convolution",
        int(output.shape[-1]),
        values,
        (
            kind,
            metadata,
            output_role,
            False,
            tuple(int(dim) for dim in operands[2].shape),
            0,
        ),
    )
    return evaluate(
        repr((tuple(program.nodes), ((root, 0, "target"),))),
        inputs,
        edge_index[0],
        edge_index[1],
        output.shape[0],
    )[0]


def materialize(radial, edge_index):
    """Gather partitioned radial inputs for the PyTorch reference."""
    if not isinstance(radial, tuple):
        return radial
    return torch.cat(
        [
            value if kind == "edge" else value[edge_index[0 if kind == "source" else 1]]
            for value, kind in radial
        ],
        dim=-1,
    )
