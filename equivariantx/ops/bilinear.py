"""Joint expert contractions with recursively transposed derivatives."""

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from ..utils.outputs import bilinear_outputs


@torch.library.custom_op("eqx::bilinear_ace", mutates_args=(), device_types="cuda")
def contract(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    from ..backends import get_kernel

    return get_kernel("bilinear_ace", "cuda")(metadata, program, node_type, operands)


contract.register_fake(bilinear_outputs)


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


contract.register_autograd(backward, setup_context=setup_context)
