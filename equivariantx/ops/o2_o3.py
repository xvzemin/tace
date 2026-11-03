"""O2 o3 operations."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from .contraction import adjoint_program


@dataclass(eq=False)
class _KernelPlan:
    """Tensor-free scheduling metadata shared by compiled calls."""

    path_data: tuple
    weight_numel: int
    has_unweighted: bool
    sparse_paths: tuple
    scalar_paths: tuple = ()
    harmonic_degrees: tuple = ()


@lru_cache(maxsize=256)
def kernel_plan(metadata):
    return _KernelPlan(*parse_metadata(metadata))


@torch.library.custom_op("eqx::contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Keep runtime scheduling and kernel compilation outside tensor tracing."""
    from ..backends import get_kernel

    return get_kernel("contraction", "cuda")(
        metadata, program, source, target, operands
    )


@contraction.register_fake
def contraction_fake(metadata, program, source, target, operands):
    program = parse_metadata(program)
    results = [None] * (1 + max(slot for _, _, pairs in program for _, slot in pairs))
    for mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


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


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)
