"""Differentiable sparse products for indexed ACE and expert schedules."""

import torch

from ....utils.metadata import decode_plans, parse_metadata
from ....utils.outputs import ace_outputs, bilinear_outputs
from ...polynomial import program_terms


def ace_contract(metadata, program, node_type, operands):
    """Evaluate one channel-product plan with indexed coefficients."""
    return contract("ace", metadata, program, node_type, operands)


def bilinear_contract(metadata, program, node_type, operands):
    """Evaluate gated product plans with element and shared coefficients."""
    return contract("bilinear", metadata, program, node_type, operands)


def channel_slice(value, start, width, stride):
    """Select a strided channel block without flattening node dimensions."""
    if not stride:
        return value[:, start : start + 1].expand(-1, width)
    return value[:, start : start + width * stride : stride]


def contract(kind, metadata, program, node_type, operands):
    """Evaluate any primal or transposed role using ordinary Torch operations."""
    plans = (
        decode_plans(metadata) if kind == "bilinear" else (parse_metadata(metadata),)
    )
    allocate = bilinear_outputs if kind == "bilinear" else ace_outputs
    zero = sum(value[..., :0].sum() for value in operands)
    outputs = [
        torch.zeros_like(value) + zero
        for value in allocate(metadata, program, node_type, operands)
    ]
    for plan, mapping, role, slot, _ in program_terms(kind, program):
        _, weighted, schedules = plans[plan]
        values = [operands[i] for i in mapping]
        for _, entries in schedules[role]:
            for record, coefficient in entries:
                ci = record[4]
                if weighted:
                    co = record[5]
                    weight = values[2]
                    types = (
                        torch.zeros_like(node_type)
                        if weight.shape[0] == 1
                        else node_type
                    )
                    if role != 2:
                        matrix = weight[:, record[2] : record[2] + ci * co]
                        matrix = matrix[types].reshape(-1, ci, co)
                    if role != 3:
                        output = channel_slice(values[3], record[3], co, record[8])
                    if role in (0, 1, 4):
                        result = (output.unsqueeze(1) @ matrix.transpose(1, 2)).squeeze(
                            1
                        )
                    else:
                        result = values[0].new_ones((node_type.shape[0], ci))
                    for factor in (0, 1, 4) if kind == "bilinear" else (0, 1):
                        if factor != role:
                            start = record[9] if factor == 4 else record[factor]
                            stride = 1 if factor == 4 else record[factor + 6]
                            result = result * channel_slice(
                                values[factor], start, ci, stride
                            )
                    if role == 3:
                        result = (result.unsqueeze(1) @ matrix).squeeze(1)
                    elif role == 2:
                        result = (result.unsqueeze(2) * output.unsqueeze(1)).flatten(1)
                        result = result.new_zeros((weight.shape[0], ci * co)).index_add(
                            0, types, result
                        )
                    start = record[9] if role == 4 else record[role]
                    stride = (
                        1
                        if role in (2, 4)
                        else record[role + 6]
                        if role < 2
                        else record[8]
                    )
                else:
                    result = values[0].new_ones((node_type.shape[0], ci))
                    for factor in range(3):
                        if factor != role:
                            result = result * channel_slice(
                                values[factor], record[factor], ci, record[factor + 6]
                            )
                    start, stride = record[role], record[role + 6]
                indices = (
                    start
                    + torch.arange(result.shape[1], device=node_type.device) * stride
                )
                outputs[slot] = outputs[slot].index_add(
                    1, indices, result * coefficient
                )
    return outputs
