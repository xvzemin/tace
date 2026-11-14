"""Sparse channel-wise products and their transposed contractions."""

import torch

from ...utils.autograd import gradient_mask
from ...utils.metadata import parse_metadata
from ...utils.outputs import ace_outputs
from .polynomial import native_polynomial


def ace_inputs(correlation, features, weight, output_dim):
    """Describe a product with an optional fused coefficient projection."""
    operands = [correlation, features]
    if weight is not None:
        operands.append(weight)
    operands.append(features.new_empty((0, output_dim)))
    program = repr(((tuple(range(len(operands))), len(operands) - 1, 0),))
    return program, operands


@torch.library.custom_op("eqx::ace_contract", mutates_args=(), device_types="cuda")
def contract_cuda(
    metadata: str,
    program: str,
    node_type: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate a multilinear contraction without expanded CG products."""
    from .. import get_kernel

    return get_kernel("ace_contract", "cuda")(metadata, program, node_type, operands)


contract_cuda.register_fake(ace_outputs)


def setup_context(ctx, inputs, output):
    metadata, program, node_type, operands = inputs
    ctx.kernel_metadata = metadata
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
    for mapping, output, slot in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and required[index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append((tuple(replacement), role, destination))
    gradients = [None] * len(operands)
    if terms:
        results = contract(ctx.kernel_metadata, repr(tuple(terms)), node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


contract_cuda.register_autograd(backward, setup_context=setup_context)


def contract(metadata, program, node_type, operands):
    """Evaluate sparse native products with analytic functional derivatives."""
    return list(native_polynomial("ace", metadata, program, node_type, *operands))
