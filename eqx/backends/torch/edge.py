"""PyTorch evaluation of edge expressions."""

from functools import lru_cache

import torch

from ...utils.metadata import parse_metadata


@lru_cache(maxsize=64)
def program_indices(metadata, device):
    """Cache fixed column indices on the operand device."""
    nodes, _ = parse_metadata(metadata)
    return {
        i: torch.tensor(data, device=device, dtype=torch.long)
        for i, (op, _, _, data) in enumerate(nodes)
        if op in ("gather", "scatter")
    }


def evaluate_torch(metadata, inputs, source, target, num_nodes):
    """Evaluate an edge program with PyTorch operations and ordinary autograd."""
    nodes, outputs = parse_metadata(metadata)
    edges = source.numel()
    indices = program_indices(metadata, inputs[0].device)
    values = []
    for node, (op, size, args, data) in enumerate(nodes):
        operands = [values[i] for i in args]
        if op == "input":
            slot, kind = data
            value = inputs[slot].flatten(1)
            if kind in ("source", "target"):
                value = value[source if kind == "source" else target]
            elif kind == "shared":
                value = value.expand(edges, -1)
        elif op == "constant":
            value = inputs[0].new_full((), data).expand(edges, size)
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
            value = operands[0].index_select(1, indices[node])
        elif op == "scatter":
            value = (
                inputs[0]
                .new_zeros((edges, size))
                .index_add(1, indices[node], operands[0])
            )
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
            for components, coefficient in paths:
                if components[role] < 0 or any(
                    index < 0
                    for k, index in enumerate(components)
                    if required & (1 << k)
                ):
                    continue
                term = None
                for k, index in enumerate(components):
                    if k != role and index >= 0:
                        factor = operands[k][:, index * width : (index + 1) * width]
                        term = factor if term is None else term * factor
                term = (
                    inputs[0].new_full((), coefficient).expand(edges, width)
                    if term is None
                    else term * coefficient
                )
                parts[components[role]].append(term)
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
    zero = sum(value[..., :0].sum() for value in inputs)
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
