"""Sparse channel polynomials using differentiable gathers and reductions."""

from collections import defaultdict
from functools import lru_cache

import torch

from ....utils.metadata import channel_product_layout, parse_metadata


@lru_cache(maxsize=128)
def contraction_groups(metadata, program):
    """Collect compatible monomials without expanding tensor operands."""
    _, rows = channel_product_layout(metadata)
    groups = defaultdict(list)
    for mapping, output, slot, required in parse_metadata(program):
        for indices, coefficient, width, strides in rows:
            if indices[output] < 0 or any(
                index < 0
                for role, index in enumerate(indices)
                if required & (1 << role)
            ):
                continue
            roles = tuple(
                role
                for role, index in enumerate(indices)
                if role != output and index >= 0
            )
            key = slot, mapping[output], tuple(mapping[role] for role in roles)
            for channel in range(width):
                groups[key].append(
                    (
                        indices[output] + channel * strides[output],
                        coefficient,
                        tuple(
                            indices[role] + channel * strides[role] for role in roles
                        ),
                    )
                )
    return tuple((key, tuple(values)) for key, values in groups.items())


def empty_outputs(metadata, program, operands):
    """Allocate connected zeros, including disconnected or empty polynomials."""
    dimensions, _ = channel_product_layout(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (1 + max(slot for _, _, slot, _ in terms))
    zero = sum(value[..., :0].sum() for value in operands)
    for mapping, role, slot, _ in terms:
        rows = 1 if operands[mapping[role]].shape[0] == 1 else operands[0].shape[0]
        if outputs[slot] is None:
            outputs[slot] = operands[0].new_zeros((rows, dimensions[role])) + zero
    return outputs


def channel_product(metadata, program, operands):
    """Evaluate grouped products and shared-operand reductions with Torch."""
    outputs = empty_outputs(metadata, program, operands)
    nodes, device = operands[0].shape[0], operands[0].device
    for (slot, output, factors), entries in contraction_groups(metadata, program):
        indices, coefficients, columns = zip(*entries)
        value = operands[0].new_tensor(coefficients).unsqueeze(0)
        for role, factor in enumerate(factors):
            index = torch.tensor([column[role] for column in columns], device=device)
            value = value * operands[factor].index_select(1, index)
        value = value.expand(nodes, -1)
        if operands[output].shape[0] == 1:
            value = value.sum(0, keepdim=True)
        index = torch.tensor(indices, device=device)
        outputs[slot] = outputs[slot].index_add(1, index, value)
    return outputs


def reference(metadata, program, operands):
    """Evaluate each coefficient and channel range independently."""
    _, rows = channel_product_layout(metadata)
    outputs = empty_outputs(metadata, program, operands)
    nodes, device = operands[0].shape[0], operands[0].device
    for mapping, output, slot, required in parse_metadata(program):
        for indices, coefficient, width, strides in rows:
            if indices[output] < 0 or any(
                index < 0
                for role, index in enumerate(indices)
                if required & (1 << role)
            ):
                continue
            channels = torch.arange(width, device=device)
            value = operands[0].new_full((1, width), coefficient)
            for role, index in enumerate(indices):
                if role != output and index >= 0:
                    value = value * operands[mapping[role]].index_select(
                        1, index + channels * strides[role]
                    )
            value = value.expand(nodes, -1)
            if operands[mapping[output]].shape[0] == 1:
                value = value.sum(0, keepdim=True)
            outputs[slot] = outputs[slot].index_add(
                1, indices[output] + channels * strides[output], value
            )
    return outputs
