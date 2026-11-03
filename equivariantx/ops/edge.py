"""Differentiable CUDA execution of static edge expressions."""

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from ..utils.outputs import edge_outputs
from ..utils.program import next_adjoint


@torch.library.custom_op("eqx::edge_program", mutates_args=(), device_types="cuda")
def evaluate(
    metadata: str,
    inputs: list[torch.Tensor],
    source: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    from ..backends import get_kernel

    return get_kernel("edge_program", "cuda")(
        metadata, inputs, source, target, num_nodes
    )


evaluate.register_fake(edge_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, values, source, target, ctx.num_nodes = inputs
    ctx.save_for_backward(source, target, *values)
    ctx.set_materialize_grads(False)


def backward(ctx, gradients):
    source, target, *inputs = ctx.saved_tensors
    slots = sorted({slot for _, slot, _ in parse_metadata(ctx.kernel_metadata)[1]})
    required = gradient_mask(inputs, ctx.needs_input_grad[1], offset=0, trailing=2)
    active = tuple(i for i, need in enumerate(required) if need)
    seeds = tuple(slot for slot, grad in zip(slots, gradients) if grad is not None)
    metadata = next_adjoint(ctx.kernel_metadata, active, seeds, len(inputs))
    adjoints = parse_metadata(metadata)[1]
    results = [None] * len(inputs)
    if adjoints:
        values = inputs + [grad for grad in gradients if grad is not None]
        grads = evaluate(metadata, values, source, target, ctx.num_nodes)
        for slot, grad in zip(sorted({x[1] for x in adjoints}), grads):
            results[slot] = grad.reshape(inputs[slot].shape)
    return None, results, None, None, None


evaluate.register_autograd(backward, setup_context=setup_context)
