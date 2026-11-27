"""Joint expert contractions with recursively transposed derivatives."""

import torch

from equivariantx.ops import get_kernel, register_kernel
from equivariantx.ops.common.polynomial import (
    EXTERNAL_POLYNOMIALS,
    native_polynomial,
    polynomial_adjoint,
)
from equivariantx.ops.common.torch.ace import (
    indexed_plan_contract,
    indexed_plan_reference,
)
from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.outputs import indexed_plan_outputs


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


@torch.library.custom_op("tace::bilinear_ace", mutates_args=(), device_types="cuda")
def contract_cuda(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    return get_kernel("bilinear_ace", "cuda")(metadata, program, node_type, operands)


contract_cuda.register_fake(indexed_plan_outputs)

register_kernel("bilinear_ace", "cuda", "tace.ops.bilinear_kernel:contract")
register_kernel(
    "bilinear_ace", "torch", indexed_plan_contract, reference=indexed_plan_reference
)
EXTERNAL_POLYNOMIALS["indexed_plan"] = (
    torch.ops.tace.bilinear_ace.default,
    indexed_plan_contract,
)


def setup_context(ctx, inputs, output):
    metadata, program, node_type, operands = inputs
    ctx.plans = metadata
    ctx.program = program
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_outputs):
    node_type, *operands = ctx.saved_tensors
    required = gradient_mask(operands, ctx.needs_input_grad[3], offset=1)
    program, values, destinations = polynomial_adjoint(
        ctx.program, operands, grad_outputs, required, kind="indexed_plan"
    )
    gradients = [None] * len(operands)
    if destinations:
        results = contract(ctx.plans, program, node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


contract_cuda.register_autograd(backward, setup_context=setup_context)


def contract(metadata, program, node_type, operands):
    """Evaluate expert products with analytic JVP and batching rules."""
    return list(
        native_polynomial("indexed_plan", metadata, program, node_type, *operands)
    )
