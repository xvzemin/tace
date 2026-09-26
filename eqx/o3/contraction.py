"""Indexed bilinear contractions and their transposes."""

import torch

from .._metadata import parse_metadata


@torch.library.custom_op("eqx::element_linear", mutates_args=(), device_types="cuda")
def contract(
    metadata: str,
    role: int,
    node_type: torch.Tensor,
    features: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Contract two operands into the selected feature or weight role."""
    from .cuda import launch

    result = fake(metadata, role, node_type, features, weight, output)
    launch(
        parse_metadata(metadata), role, node_type, (features, weight, output), result
    )
    return result


@contract.register_fake
def fake(metadata, role, node_type, features, weight, output):
    din, dout, _, _, _ = parse_metadata(metadata)
    shape = (
        weight.shape if role == 1 else (node_type.shape[0], din if role == 0 else dout)
    )
    return features.new_empty(shape)


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


contract.register_autograd(backward, setup_context=setup_context)
