"""Radial networks evaluated inside tiled convolution programs."""

import torch

from ..utils.metadata import parse_metadata
from .program import Program


def compose_radial(metadata, inputs, slot, network):
    """Replace an edge input by a tiled radial network evaluation."""
    from ..models.radial import radial_program

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
    from ..models.radial import radial_program
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
