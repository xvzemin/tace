"""Differentiable CUDA execution of static edge expressions."""

from functools import lru_cache

import torch

from ..utils.metadata import parse_metadata
from .contraction import gradient_mask
from .program import next_adjoint


def evaluate_torch(metadata, inputs, source, target, num_nodes):
    """Evaluate an edge program with PyTorch operations and ordinary autograd."""
    nodes, outputs = parse_metadata(metadata)
    edges = source.numel()
    values = []
    for op, size, args, data in nodes:
        operands = [values[i] for i in args]
        if op == "input":
            slot, kind = data
            value = inputs[slot].flatten(1)
            if kind in ("source", "target"):
                value = value[source if kind == "source" else target]
            elif kind == "shared":
                value = value.expand(edges, -1)
        elif op == "constant":
            value = inputs[0].new_full((edges, size), data)
        elif op == "add":
            value = operands[0] + operands[1]
        elif op == "mul":
            value = operands[0] * operands[1]
        elif op in ("normalize", "inv_norm"):
            x = operands[0]
            if data[1]:
                x = x - x.mean(-1, keepdim=True)
            inverse = (x.square().mean(-1, keepdim=True) + data[0]).rsqrt()
            value = x * inverse if op == "normalize" else inverse.expand_as(x)
        elif op == "mean":
            value = operands[0].mean(-1, keepdim=True).expand_as(operands[0])
        elif op in ("exp", "sin", "cos", "tanh", "sigmoid", "reciprocal", "rsqrt"):
            value = getattr(torch, op)(operands[0])
        elif op == "silu":
            value = torch.nn.functional.silu(operands[0])
        elif op == "slice":
            value = operands[0][:, data : data + size]
        elif op == "concat":
            value = (
                torch.cat(operands, dim=-1)
                if operands
                else inputs[0].new_empty((edges, 0))
            )
        elif op == "gather":
            value = operands[0][:, list(data)]
        elif op == "scatter":
            index = torch.tensor(data, device=inputs[0].device, dtype=torch.long)
            value = inputs[0].new_zeros((edges, size)).index_add(1, index, operands[0])
        elif op == "transpose":
            rows, columns = data
            value = operands[0].reshape(edges, rows, columns).transpose(1, 2).flatten(1)
        elif op == "matmul":
            rows, width, columns = data
            value = torch.bmm(
                operands[0].reshape(edges, rows, width),
                operands[1].reshape(edges, width, columns),
            ).flatten(1)
        elif op == "product":
            (width, dims, paths), role, required = data
            parts = [[] for _ in range(dims[role])]
            for indices, coefficient in paths:
                if indices[role] < 0 or any(
                    index < 0 for k, index in enumerate(indices) if required & (1 << k)
                ):
                    continue
                term = inputs[0].new_full((edges, width), coefficient)
                for k, index in enumerate(indices):
                    if k != role and index >= 0:
                        term = (
                            term * operands[k][:, index * width : (index + 1) * width]
                        )
                parts[indices[role]].append(term)
            value = torch.cat(
                [
                    sum(part[1:], part[0])
                    if part
                    else inputs[0].new_zeros((edges, width))
                    for part in parts
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError(f"PyTorch edge operation {op!r} is unavailable.")
        values.append(value)
    result = {}
    zero = sum(value.sum() * 0 for value in inputs)
    for root, slot, kind in outputs:
        value = values[root] + zero
        if kind in ("source", "target"):
            value = value.new_zeros((num_nodes, value.shape[1])).index_add(
                0, source if kind == "source" else target, value
            )
        elif kind == "shared":
            value = value.sum(0, keepdim=True)
        result[slot] = result[slot] + value if slot in result else value
    return [result[slot] for slot in sorted(result)]


@torch.library.custom_op("eqx::edge_program", mutates_args=(), device_types="cuda")
def evaluate(
    metadata: str,
    inputs: list[torch.Tensor],
    source: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    from .codegen import launch
    from .execution import launch as launch_tiled
    from .execution import uses_matrix_products

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
        if writes != ["edge"] or not source.numel() or uses_matrix_products(metadata):
            output.zero_()
    if source.numel() and outputs:
        if uses_matrix_products(metadata):
            launch = launch_tiled
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
