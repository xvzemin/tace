"""Symmetric polynomials and recursive multilinear adjoints."""

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from ..utils.outputs import symmetric_outputs
from .polynomial import native_polynomial, polynomial_adjoint


@torch.library.custom_op(
    "eqx::symmetric_contract", mutates_args=(), device_types="cuda"
)
def symmetric_contract_cuda(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate element-weighted symmetric contractions."""
    from . import get_kernel

    return get_kernel("symmetric_contract", "cuda")(
        metadata, program, node_type, operands
    )


symmetric_contract_cuda.register_fake(symmetric_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, node_type, operands = inputs
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_outputs):
    node_type, *operands = ctx.saved_tensors
    needed = gradient_mask(operands, ctx.needs_input_grad[3], offset=1)
    program, values, destinations = polynomial_adjoint(
        repr(ctx.program), operands, grad_outputs, needed
    )
    gradients = [None] * len(operands)
    if destinations:
        results = symmetric_contract(ctx.kernel_metadata, program, node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


symmetric_contract_cuda.register_autograd(backward, setup_context=setup_context)


def symmetric_contract(metadata, program, node_type, operands):
    """Evaluate native polynomials with explicit JVP and batching rules."""
    return list(native_polynomial("symmetric", metadata, program, node_type, *operands))


from .decompositions import register_symmetric_decomposition

register_symmetric_decomposition()
