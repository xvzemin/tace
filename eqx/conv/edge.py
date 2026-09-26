"""Differentiable CUDA execution of static edge expressions."""

from functools import lru_cache

import torch

from .._metadata import parse_metadata
from .program import next_adjoint


@torch.library.custom_op("eqx::edge_program", mutates_args=(), device_types="cuda")
def evaluate(
    metadata: str,
    inputs: list[torch.Tensor],
    source: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    from .codegen import launch

    if inputs[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Fused edge expressions require float32 or float64.")
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("Source and target must be one-dimensional edge indices.")
    for slot, kind, width in input_specifications(metadata):
        rows = (
            1 if kind == "shared" else source.numel() if kind == "edge" else num_nodes
        )
        if inputs[slot].numel() != rows * width:
            raise ValueError("Input shape does not match the fused expression layout.")
    inputs = [x.contiguous() for x in inputs]
    outputs = evaluate_fake(metadata, inputs, source, target, num_nodes)
    descriptions = parse_metadata(metadata)[1]
    slots = sorted({slot for _, slot, _ in descriptions})
    for slot, output in zip(slots, outputs):
        writes = [kind for _, s, kind in descriptions if s == slot]
        if writes != ["edge"] or not source.numel():
            output.zero_()
    if source.numel() and outputs:
        launch(metadata, inputs, source.contiguous(), target.contiguous(), outputs)
    return outputs


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
    active = tuple(i for i, need in enumerate(ctx.needs_input_grad[1]) if need)
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
