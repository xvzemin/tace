"""Indexed bilinear contractions and their transposes."""

import torch

from ..utils.outputs import linear_output
from .polynomial import native_polynomial


@torch.library.custom_op("eqx::element_linear", mutates_args=(), device_types="cuda")
def contract_cuda(
    metadata: str,
    role: int,
    node_type: torch.Tensor,
    features: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Contract two operands into the selected feature or weight role."""
    from . import get_kernel

    return get_kernel("element_linear", "cuda")(
        metadata, role, node_type, features, weight, output
    )


contract_cuda.register_fake(linear_output)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, ctx.role, node_type, *operands = inputs
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_output):
    node_type, *operands = ctx.saved_tensors
    operands[ctx.role] = grad_output
    gradients = [None, None, None]
    for role in range(3):
        if role != ctx.role and ctx.needs_input_grad[role + 3]:
            gradients[role] = contract(ctx.kernel_metadata, role, node_type, *operands)
    return None, None, None, *gradients


contract_cuda.register_autograd(backward, setup_context=setup_context)


def contract(metadata, role, node_type, features, weight, output):
    """Apply native indexed maps with explicit functional-transform rules."""
    program = repr((((0, 1, 2), role, 0),))
    return native_polynomial(
        "linear", metadata, program, node_type, features, weight, output
    )[0]
