"""Symmetric polynomials and recursive multilinear adjoints."""

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata


@torch.library.custom_op(
    "eqx::symmetric_contract", mutates_args=(), device_types="cuda"
)
def symmetric_contract(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate element-weighted symmetric contractions."""
    from ..backends import get_kernel

    return get_kernel("symmetric_contract", "cuda")(
        metadata, program, node_type, operands
    )


@symmetric_contract.register_fake
def symmetric_contract_fake(metadata, program, node_type, operands):
    input_dim, paths, output_dim, degree, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (max(term[2] for term in terms) + 1)
    for mapping, role, slot, _ in terms:
        if outputs[slot] is None:
            size = (
                paths
                if role == degree
                else output_dim
                if role == degree + 1
                else input_dim
            )
            nodes = (
                operands[mapping[role]].shape[0]
                if role == degree
                else node_type.shape[0]
            )
            outputs[slot] = operands[0].new_empty((nodes, size, operands[0].shape[-1]))
    return outputs


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, node_type, operands = inputs
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_outputs):
    node_type, *operands = ctx.saved_tensors
    values, cotangents = list(operands), {}
    for slot, gradient in enumerate(grad_outputs):
        if gradient is not None:
            cotangents[slot] = len(values)
            values.append(gradient)
    terms, destinations = [], {}
    needed = gradient_mask(operands, ctx.needs_input_grad[3], offset=1)
    for mapping, output, slot, required in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and needed[index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append(
                    (tuple(replacement), role, destination, required | (1 << output))
                )
    gradients = [None] * len(operands)
    if terms:
        results = symmetric_contract(
            ctx.kernel_metadata, repr(tuple(terms)), node_type, values
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


symmetric_contract.register_autograd(backward, setup_context=setup_context)
