"""Streaming TECE interactions and derivatives."""

import torch

from ...utils.autograd import gradient_mask
from ...utils.metadata import parse_metadata
from ...utils.outputs import tece_contraction_outputs, tece_outputs
from ...utils.program import next_adjoint


@torch.library.custom_op("eqx::tece_interaction", mutates_args=(), device_types="cuda")
def interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    from .. import get_kernel

    return get_kernel("tece_interaction", "cuda")(metadata, source, target, inputs)


interaction.register_fake(tece_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, source, target, values = inputs
    ctx.save_for_backward(source, target, *values, *output)
    ctx.num_inputs = len(values)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def backward(ctx, gradients):
    from ...models.tace.tece_oam_rra.expressions import base_program, first_adjoint

    source, target, *saved = ctx.saved_tensors
    values = saved[: ctx.num_inputs]
    result, denominator, maximum = saved[ctx.num_inputs :]
    grad_result, grad_denominator, _ = gradients
    grad_result = torch.zeros_like(result) if grad_result is None else grad_result
    grad_denominator = (
        torch.zeros_like(denominator) if grad_denominator is None else grad_denominator
    )
    required = gradient_mask(values, ctx.needs_input_grad[3])
    active = tuple(i for i, need in enumerate(required) if need)
    output = [None] * ctx.num_inputs
    if active:
        metadata = first_adjoint(base_program(ctx.kernel_metadata, values), active)
        inputs = [*values, result, denominator, maximum, grad_result, grad_denominator]
        slots = tuple(sorted({slot for _, slot, _ in parse_metadata(metadata)[1]}))
        for slot, value in zip(slots, contraction(metadata, source, target, inputs)):
            output[slot] = value
    return None, None, None, output


interaction.register_autograd(backward, setup_context=setup_context)


@torch.library.custom_op("eqx::tece_contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate local expressions and their derivatives with CUDA kernels."""
    from .. import get_kernel

    return get_kernel("tece_contraction", "cuda")(metadata, source, target, inputs)


contraction.register_fake(tece_contraction_outputs)


def contraction_setup_context(ctx, inputs, output):
    ctx.kernel_metadata, source, target, values = inputs
    ctx.save_for_backward(source, target, *values)
    ctx.set_materialize_grads(False)


def contraction_backward(ctx, gradients):
    source, target, *inputs = ctx.saved_tensors
    slots = sorted({slot for _, slot, _ in parse_metadata(ctx.kernel_metadata)[1]})
    seeds = tuple(slot for slot, grad in zip(slots, gradients) if grad is not None)
    required = gradient_mask(inputs, ctx.needs_input_grad[3])
    active = tuple(i for i, need in enumerate(required) if need)
    result = [None] * len(inputs)
    if active and seeds:
        metadata = next_adjoint(ctx.kernel_metadata, active, seeds, len(inputs))
        slots_out = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
        values = contraction(
            metadata,
            source,
            target,
            inputs + [grad for grad in gradients if grad is not None],
        )
        for slot, value in zip(slots_out, values):
            result[slot] = value
    return None, None, None, result


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)
