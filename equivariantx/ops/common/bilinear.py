"""Joint expert contractions with recursively transposed derivatives."""

import torch

from ...utils.autograd import gradient_mask
from ...utils.metadata import parse_metadata
from ...utils.outputs import bilinear_outputs
from .polynomial import native_polynomial


def bilinear_inputs(features, base, gates, weight, output_dim, shared_weight=None):
    """Describe element/expert and optional shared coefficient contributions."""
    operands = [
        features,
        base,
        weight.flatten(1),
        features.new_empty((0, output_dim)),
        gates,
    ]
    program = [(0, (0, 1, 2, 3, 4), 3, 0)]
    if shared_weight is not None:
        operands.append(shared_weight.reshape(1, -1))
        program.append((1, (0, 1, 5, 3, 4), 3, 0))
    return repr(tuple(program)), operands


@torch.library.custom_op("eqx::bilinear_ace", mutates_args=(), device_types="cuda")
def contract_cuda(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    from .. import get_kernel

    return get_kernel("bilinear_ace", "cuda")(metadata, program, node_type, operands)


contract_cuda.register_fake(bilinear_outputs)


def setup_context(ctx, inputs, output):
    metadata, program, node_type, operands = inputs
    ctx.plans = metadata
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_outputs):
    node_type, *operands = ctx.saved_tensors
    values = list(operands)
    cotangents = {}
    for slot, grad in enumerate(grad_outputs):
        if grad is not None:
            cotangents[slot] = len(values)
            values.append(grad)
    terms, destinations = [], {}
    required = gradient_mask(operands, ctx.needs_input_grad[3], offset=1)
    for plan, mapping, output, slot in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and required[index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append((plan, tuple(replacement), role, destination))
    gradients = [None] * len(operands)
    if terms:
        results = contract(ctx.plans, repr(tuple(terms)), node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


contract_cuda.register_autograd(backward, setup_context=setup_context)


def contract(metadata, program, node_type, operands):
    """Evaluate expert products with analytic JVP and batching rules."""
    return list(native_polynomial("bilinear", metadata, program, node_type, *operands))
