"""Degree-wise frame rotations and their recursive transposes."""

import torch

from ..utils.outputs import rotation_output
from .precision import validate_uniform


def rotate(features, matrix, metadata):
    """Multiply frames within matching degrees without dense angular work."""
    matrix = matrix.to(features)
    empty = features.new_empty((0, matrix.shape[1], features.shape[2]))
    return contract(features, matrix, empty, metadata, 2)


@torch.library.custom_op("eqx::frame_rotation", mutates_args=(), device_types="cuda")
def contract_cuda(
    features: torch.Tensor,
    matrix: torch.Tensor,
    edge: torch.Tensor,
    metadata: str,
    role: int,
) -> torch.Tensor:
    from . import get_kernel

    return get_kernel("frame_rotation", "cuda")(features, matrix, edge, metadata, role)


contract_cuda.register_fake(rotation_output)


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


contract_cuda.register_autograd(backward, setup_context=setup_context)


@torch.compiler.allow_in_graph
def contract(features, matrix, edge, metadata, role):
    """Expose degree-wise products inside functional transformations."""
    validate_uniform((features, matrix, edge))
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.frame import frame_rotation

        return frame_rotation(features, matrix, edge, metadata, role)
    return torch.ops.eqx.frame_rotation(features, matrix, edge, metadata, role)
