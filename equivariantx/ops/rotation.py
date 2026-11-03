"""Degree-wise frame rotations and their recursive transposes."""

import torch


def rotate(features, matrix, metadata):
    """Multiply frames within matching degrees without dense angular work."""
    empty = features.new_empty((0, matrix.shape[1], features.shape[2]))
    return contract(features, matrix, empty, metadata, 2)


@torch.library.custom_op("eqx::frame_rotation", mutates_args=(), device_types="cuda")
def contract(
    features: torch.Tensor,
    matrix: torch.Tensor,
    edge: torch.Tensor,
    metadata: str,
    role: int,
) -> torch.Tensor:
    from ..backends import get_kernel

    return get_kernel("frame_rotation", "cuda")(features, matrix, edge, metadata, role)


@contract.register_fake
def contract_fake(features, matrix, edge, metadata, role):
    if role == 0:
        return torch.empty_like(features)
    if role == 1:
        return torch.empty_like(matrix)
    return features.new_empty((features.shape[0], matrix.shape[1], features.shape[2]))


def setup_context(ctx, inputs, output):
    features, matrix, edge, ctx.kernel_metadata, ctx.role = inputs
    ctx.save_for_backward(features, matrix, edge)


def backward(ctx, grad):
    values = list(ctx.saved_tensors)
    values[ctx.role] = grad
    return (
        *[
            contract(*values, ctx.kernel_metadata, role)
            if role != ctx.role and ctx.needs_input_grad[role]
            else None
            for role in range(3)
        ],
        None,
        None,
    )


contract.register_autograd(backward, setup_context=setup_context)
