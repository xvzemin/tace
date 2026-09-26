"""Sparse channel-wise products and their transposed contractions."""

from ast import literal_eval
from functools import lru_cache

import torch

parse = lru_cache(maxsize=256)(literal_eval)


@torch.library.custom_op("eqx::ace_contract", mutates_args=(), device_types="cuda")
def contract(
    metadata: str,
    program: str,
    node_type: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate a multilinear contraction without expanded CG products."""
    from ..kernels.cuda_graph import execute
    from .cuda import launch

    outputs = contract_fake(metadata, program, node_type, operands)
    terms = parse(program)

    def run(inputs, outputs):
        launch(parse(metadata), terms, inputs[0], inputs[1:], outputs)

    read = {0}
    for mapping, output, _ in terms:
        read.update(1 + index for role, index in enumerate(mapping) if role != output)
    execute("ace", (metadata, program), run, (node_type, *operands), outputs, read=read)
    return outputs


@contract.register_fake
def contract_fake(metadata, program, node_type, operands):
    dims, weighted, _ = parse(metadata)
    terms = parse(program)
    outputs = [None] * (max(slot for _, _, slot in terms) + 1)
    for mapping, role, slot in terms:
        if outputs[slot] is None:
            rows = (
                operands[mapping[2]].shape[0]
                if weighted and role == 2
                else node_type.shape[0]
            )
            outputs[slot] = operands[0].new_empty((rows, dims[role]))
    return outputs


def setup_context(ctx, inputs, output):
    metadata, program, node_type, operands = inputs
    ctx.kernel_metadata = metadata
    ctx.program = parse(program)
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
    for mapping, output, slot in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and ctx.needs_input_grad[3][index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append((tuple(replacement), role, destination))
    gradients = [None] * len(operands)
    if terms:
        results = contract(ctx.kernel_metadata, repr(tuple(terms)), node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


contract.register_autograd(backward, setup_context=setup_context)
