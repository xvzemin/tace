"""O2 o3 operations."""

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import kernel_plan, parse_metadata
from ..utils.outputs import contraction_outputs
from .contraction import adjoint_program
from .precision import validate_uniform


@torch.library.custom_op("eqx::contraction", mutates_args=(), device_types="cuda")
def contraction_cuda(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Keep runtime scheduling and kernel compilation outside tensor tracing."""
    from . import get_kernel

    return get_kernel("contraction", "cuda")(
        metadata, program, source, target, operands
    )


@contraction_cuda.register_fake
def contraction_fake(metadata, program, source, target, operands):
    return contraction_outputs(program, operands)


def contraction_setup_context(ctx, inputs, output):
    metadata, program, source, target, operands = inputs
    ctx.kernel_metadata = metadata
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def contraction_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        gradient_mask(operands, ctx.needs_input_grad[4]),
        kernel_plan(ctx.kernel_metadata).has_unweighted,
    )
    gradients = [None] * len(operands)
    if program:
        results = contraction(
            ctx.kernel_metadata,
            repr(program),
            source,
            target,
            values,
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, None, gradients


contraction_cuda.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)


@torch.compiler.allow_in_graph
def contraction(metadata, program, source, target, operands):
    """Expose frame-convolution transposes inside functional transformations."""
    validate_uniform(operands)
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.convolution import contraction as torch_contraction

        return torch_contraction(metadata, program, source, target, operands)
    return torch.ops.eqx.contraction(metadata, program, source, target, operands)
