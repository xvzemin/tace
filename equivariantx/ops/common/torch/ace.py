"""Differentiable sparse products for indexed ACE and expert schedules."""

import torch

from ....utils.metadata import decode_plans, parse_metadata
from ....utils.outputs import ace_outputs, bilinear_outputs
from ... import config
from ..polynomial import program_terms
from .grouped import element_grouping, element_path
from .linear import grouped_linear_available


def ace_contract(metadata, program, node_type, operands):
    """Evaluate one channel-product plan with indexed coefficients."""
    return contract("ace", metadata, program, node_type, operands)


def bilinear_contract(metadata, program, node_type, operands):
    """Evaluate gated product plans with element and shared coefficients."""
    return contract("bilinear", metadata, program, node_type, operands)


def ace_reference(metadata, program, node_type, operands):
    """Evaluate independent ACE entries with ordinary Torch operations."""
    return contract("ace", metadata, program, node_type, operands, optimized=False)


def bilinear_reference(metadata, program, node_type, operands):
    """Evaluate independent gated entries with ordinary Torch operations."""
    return contract("bilinear", metadata, program, node_type, operands, optimized=False)


def channel_slice(value, start, width, stride):
    """Select a strided channel block without flattening node dimensions."""
    if not stride:
        return value[:, start : start + 1].expand(-1, width)
    return value[:, start : start + width * stride : stride]


def coefficient_grouping(node_type, elements, groupings):
    """Reuse bounded coefficient-row ownership across independent paths."""
    nodes = node_type.shape[0]
    width = 128 if nodes >= 64 * elements else 32 if nodes >= 16 * elements else 8
    key = elements, width
    if key not in groupings:
        groupings[key] = element_grouping(node_type, elements, 1, width)
    return groupings[key], width


def indexed_product(features, matrix, node_type, groupings):
    """Apply shared or bounded element coefficient matrices."""
    elements, ci, co = matrix.shape
    if elements == 1:
        return features @ matrix[0]
    path = (0, 0, 0, ci, co, 1, 1.0)
    weight = matrix.reshape(elements, 1, ci * co)
    if not grouped_linear_available(features, weight, (path,)):
        return torch.bmm(features[:, None, :], matrix[node_type]).squeeze(1)
    grouping, width = coefficient_grouping(node_type, elements, groupings)
    return element_path(features, weight, grouping, path, "mul_ir", width)


def indexed_outer(left, right, matrix, node_type, groupings):
    """Reduce coefficient adjoints without storing one matrix per node."""
    elements, ci, co = matrix.shape
    if elements == 1:
        return (left.T @ right).unsqueeze(0)
    path = (0, 0, 0, ci, co, 1, 1.0)
    if not grouped_linear_available(
        left, matrix.reshape(elements, 1, ci * co), (path,)
    ):
        outer = left[:, :, None] * right[:, None, :]
        return outer.new_zeros((elements, ci, co)).index_add(0, node_type, outer)
    (slots, owners, chunks), width = coefficient_grouping(
        node_type, elements, groupings
    )
    left = left.new_zeros((chunks * width, ci)).index_copy(0, slots, left)
    right = right.new_zeros((chunks * width, co)).index_copy(0, slots, right)
    outer = left.reshape(chunks, width, ci).transpose(1, 2) @ right.reshape(
        chunks, width, co
    )
    return outer.new_zeros((elements, ci, co)).index_add(0, owners, outer)


def accumulated_product(kind, plans, terms, node_type, operands, outputs):
    """Share indexed coefficient products across all transposed roles."""
    groupings, projected = {}, {}
    for plan, mapping, role, slot, _ in terms:
        values = [operands[i] for i in mapping]
        weight = values[2]
        blocks = {}
        for _, entries in plans[plan][2][role]:
            for record, coefficient in entries:
                ci, co = record[4:6]
                if role in (0, 1, 4):
                    key = (
                        mapping[2],
                        mapping[3],
                        record[2],
                        record[3],
                        ci,
                        co,
                        record[8],
                    )
                    if key not in projected:
                        matrix = weight[:, record[2] : record[2] + ci * co].reshape(
                            -1, ci, co
                        )
                        output = channel_slice(values[3], record[3], co, record[8])
                        projected[key] = indexed_product(
                            output, matrix.transpose(1, 2), node_type, groupings
                        )
                    result = projected[key]
                else:
                    result = values[0].new_ones((node_type.shape[0], ci))
                for factor in (0, 1, 4) if kind == "bilinear" else (0, 1):
                    if factor != role:
                        start = record[9] if factor == 4 else record[factor]
                        stride = 1 if factor == 4 else record[factor + 6]
                        result = result * channel_slice(
                            values[factor], start, ci, stride
                        )
                key = (
                    (record[2], record[3], ci, co, record[8])
                    if role in (2, 3)
                    else (
                        record[9] if role == 4 else record[role],
                        ci,
                        1 if role == 4 else record[role + 6],
                    )
                )
                result = result * coefficient
                blocks[key] = blocks[key] + result if key in blocks else result
        for key, value in blocks.items():
            if role in (2, 3):
                sw, sy, ci, co, stride = key
                matrix = weight[:, sw : sw + ci * co].reshape(-1, ci, co)
                if role == 2:
                    output = channel_slice(values[3], sy, co, stride)
                    result = indexed_outer(
                        value, output, matrix, node_type, groupings
                    ).flatten(1)
                    start, size, stride = sw, ci * co, 1
                else:
                    result = indexed_product(value, matrix, node_type, groupings)
                    start, size = sy, co
            else:
                start, size, stride = key
                result = value
            index = start + torch.arange(size, device=node_type.device) * stride
            outputs[slot] = outputs[slot].index_add(1, index, result)
    return outputs


def contract(kind, metadata, program, node_type, operands, *, optimized=None):
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
    terms = program_terms(kind, program)
    optimized = not config.TORCH_REFERENCE if optimized is None else optimized
    if optimized and all(plans[plan][1] for plan, _, _, _, _ in terms):
        return accumulated_product(kind, plans, terms, node_type, operands, outputs)
    for plan, mapping, role, slot, _ in terms:
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
