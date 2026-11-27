"""Receiver-normalized local interactions in ordinary and tiled PyTorch."""

from functools import lru_cache

import torch

from equivariantx.ops.common.torch.edge import evaluate_recomputed, evaluate_torch
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.program import Program

from ..tece import receiver_rows


@lru_cache(maxsize=64)
def plans(base):
    nodes, score, value, specs, heads, channels, _ = base
    roots = repr((nodes, ((score, 0, "edge"), (value, 1, "edge"))))
    scores = repr((nodes, ((score, 0, "edge"),)))
    program = Program(nodes)
    weights = program.input(len(specs), "edge", heads)
    indices = tuple(
        (i % channels) // (channels // heads) for i in range(program.size(value))
    )
    message = program.binary("mul", value, program.gather(weights, indices))
    reduction = repr((tuple(program.nodes), ((message, 0, "target"),)))
    return roots, scores, reduction


def matrices(inputs, specs):
    return [
        value.reshape(1 if spec[1] == "shared" else value.shape[0], spec[0])
        if spec is not None and spec[1] != "row_map"
        else value
        for value, spec in zip(inputs, specs)
    ]


def normalized_weights(scores, cutoff, target, count, eps):
    maximum = scores.new_full((count, scores.shape[1]), -torch.inf).scatter_reduce(
        0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
    )
    maximum = torch.where(torch.isneginf(maximum), 0, maximum)
    exponential = (scores - maximum[target]).exp()
    denominator = (
        torch.zeros_like(maximum).index_add(0, target, exponential * cutoff) + eps
    )
    return exponential / denominator[target] * cutoff.square(), denominator, maximum


def forward(metadata, source, target, inputs, *, reference):
    from ...models.tace.tece_oam_rra.expressions import base_program

    base = base_program(metadata, inputs)
    nodes, _, value, specs, heads, channels, eps = base
    count, angular = receiver_rows(metadata, inputs), nodes[value][1] // channels
    values = matrices(inputs, specs)
    roots, scores, reduction = plans(base)
    if reference:
        scores, messages = evaluate_torch(roots, values, source, target, count)
    else:
        scores = evaluate_recomputed(scores, values, source, target, count)[0]
    weights, denominator, maximum = normalized_weights(
        scores, values[4], target, count, eps
    )
    if reference:
        messages = (
            messages.reshape(-1, angular, heads, channels // heads)
            * weights[:, None, :, None]
        )
        result = messages.new_zeros((count, angular, channels)).index_add(
            0, target, messages.flatten(-2)
        )
    else:
        result = evaluate_recomputed(
            reduction, [*values, weights], source, target, count
        )[0]
    return [result.reshape(count, angular, channels), denominator, maximum]


def interaction(metadata, source, target, inputs):
    """Recompute score and weighted-message tiles without retaining edge messages."""
    return forward(metadata, source, target, inputs, reference=False)


def interaction_reference(metadata, source, target, inputs):
    """Materialize scores and edge messages using ordinary autograd."""
    return forward(metadata, source, target, inputs, reference=True)


@lru_cache(maxsize=128)
def contraction_layout(metadata, num_inputs):
    from ...models.tace.tece_oam_rra.expressions import operand_specs

    nodes, outputs = parse_metadata(metadata)
    specs = operand_specs(nodes, outputs, [None] * num_inputs)
    # A derivative expression can request targets without reading them
    # (e.g. the adjoint of a purely linear term); its destination slot then
    # still carries the correct receiver extent.
    target_slot = next(
        (i for i, s in enumerate(specs) if s is not None and s[1] == "target"),
        next((slot for _, slot, kind in outputs if kind == "target"), 0),
    )
    return specs, tuple(sorted({slot for _, slot, _ in outputs})), target_slot


def contract(metadata, source, target, inputs, evaluate):
    specs, slots, target_slot = contraction_layout(metadata, len(inputs))
    outputs = evaluate(
        metadata, matrices(inputs, specs), source, target, inputs[target_slot].shape[0]
    )
    return [value.reshape_as(inputs[i]) for i, value in zip(slots, outputs)]


def contraction(metadata, source, target, inputs):
    """Evaluate requested derivative slots with the bounded edge program."""
    return contract(metadata, source, target, inputs, evaluate_recomputed)


def contraction_reference(metadata, source, target, inputs):
    """Evaluate derivative expressions with ordinary autograd."""
    return contract(metadata, source, target, inputs, evaluate_torch)
