"""Differentiable CUDA execution of static edge expressions."""

from functools import lru_cache

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from .program import next_adjoint


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


@lru_cache(maxsize=256)
def execution_mode(metadata):
    """Select element-wise execution or a narrow node reduction."""
    nodes, outputs = parse_metadata(metadata)
    if any(
        op in ("product", "matmul", "normalize", "inv_norm", "mean", "rsqrt")
        or op == "scatter"
        and nodes[args[0]][1] > 32
        for op, _, args, _ in nodes
    ):
        return "edge"
    if all(kind == "edge" for _, _, kind in outputs):
        return "pointwise"
    if len(outputs) == 1 and outputs[0][2] in ("source", "target"):
        if nodes[outputs[0][0]][1] <= 32:
            return "node"
    return "edge"


@lru_cache(maxsize=256)
def input_specifications(metadata):
    return tuple(
        (data[0], data[1], size)
        for op, size, _, data in parse_metadata(metadata)[0]
        if op == "input"
    )


@evaluate.register_fake
def evaluate_fake(metadata, inputs, source, target, num_nodes):
    nodes, outputs = parse_metadata(metadata)
    specifications = {slot: (nodes[root][1], kind) for root, slot, kind in outputs}
    return [
        inputs[0].new_empty(
            (
                1
                if kind == "shared"
                else source.numel()
                if kind == "edge"
                else num_nodes
            ),
            width,
        )
        for _, (width, kind) in sorted(specifications.items())
    ]


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
