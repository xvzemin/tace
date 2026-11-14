"""Transpose multilinear convolution programs without truncating derivatives."""

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ...utils.metadata import parse_metadata
from .. import config


def project_radial(program, operands, output_role):
    """Share bounded wide projections before differentiating a primal call."""
    if not config.RADIAL_PROJECT_MIN_WIDTH or not torch.is_grad_enabled():
        return program, operands
    terms = parse_metadata(program)
    if len(terms) != 1:
        return program, operands
    mapping, weighted, pairs = terms[0]
    radial, projection = (operands[index] for index in mapping[1:3])
    if not (
        pairs == ((output_role, 0),)
        and len(mapping) <= output_role + 2
        and statically_known_true(
            projection.shape[0] >= config.RADIAL_PROJECT_MIN_WIDTH
        )
        and statically_known_true(
            radial.shape[0] * projection.shape[-1] * radial.element_size()
            <= config.RADIAL_PROJECT_MAX_BYTES
        )
    ):
        return program, operands
    # Accumulate cotangents before the GEMMs and retain shared frame operands.
    values = [radial @ projection, projection.new_empty((0, projection.shape[-1]))]
    indices, remapped = {}, []
    for role, index in enumerate(mapping):
        if role in (1, 2):
            remapped.append(role - 1)
        else:
            if index not in indices:
                indices[index] = len(values)
                values.append(operands[index])
            remapped.append(indices[index])
    return repr(((tuple(remapped), weighted, pairs),)), values


def adjoint_program(program, operands, grad_outputs, needs_grad, has_unweighted):
    """Transpose requested outputs, retaining the dependencies of each path."""
    values = list(operands)
    cotangents = {}
    for slot, value in enumerate(grad_outputs):
        if value is not None:
            cotangents[slot] = len(values)
            values.append(value)
    terms = {}
    destinations = {}
    for mapping, weighted_only, pairs in program:
        for output, slot in pairs:
            if slot not in cotangents:
                continue
            if output == 2 and not operands[mapping[2]].numel():
                continue
            replacement = list(mapping)
            replacement[output] = cotangents[slot]
            # Roles one and two are radial features and their projection.
            key = (
                tuple(replacement),
                has_unweighted and (weighted_only or output in (1, 2)),
            )
            for role, index in enumerate(mapping):
                if role != output and needs_grad[index]:
                    destination = destinations.setdefault(index, len(destinations))
                    terms.setdefault(key, []).append((role, destination))
    program = tuple(
        (mapping, weighted, tuple(pairs))
        for (mapping, weighted), pairs in terms.items()
    )
    return program, values, destinations
