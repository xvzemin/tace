"""Streaming TECE interactions and derivatives."""

import torch

from equivariantx.ops import get_kernel
from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.program import next_adjoint


def receiver_rows(metadata, inputs):
    """Read the destination extent without specializing its runtime size.

    A compact program encodes (architecture, expression, receiver_slot).
    Both legacy forms (the raw architecture or its expression pair) use
    slot zero for both endpoints.
    """
    description = parse_metadata(metadata)
    return inputs[description[2] if len(description) == 3 else 0].shape[0]


def tece_outputs(metadata, source, target, inputs):
    description = parse_metadata(metadata)
    if len(description) in (2, 3):
        description = parse_metadata(description[0])
    channels, heads = description[2:4]
    nodes = receiver_rows(metadata, inputs)
    return [
        inputs[0].new_empty(
            (
                nodes,
                (description[0] + 1) ** 2
                if inputs[5].ndim == 2
                else inputs[6].shape[1],
                channels,
            )
        ),
        inputs[0].new_empty((nodes, heads)),
        inputs[0].new_empty((nodes, heads)),
    ]


def tece_contraction_outputs(metadata, source, target, inputs):
    slots = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
    return [
        torch.empty_like(inputs[slot], memory_format=torch.contiguous_format)
        for slot in slots
    ]


@torch.library.custom_op(
    "eqx_foundation::tece_interaction", mutates_args=(), device_types="cuda"
)
def interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    return get_kernel("tece_interaction", "cuda")(metadata, source, target, inputs)


interaction.register_fake(tece_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, source, target, values = inputs
    ctx.save_for_backward(source, target, *values, *output)
    ctx.num_inputs = len(values)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def backward(ctx, gradients):
    from ..models.tace.tece_oam_rra.expressions import base_program, first_adjoint

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


@torch.library.custom_op(
    "eqx_foundation::tece_contraction", mutates_args=(), device_types="cuda"
)
def contraction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate local expressions and their derivatives with CUDA kernels."""
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
