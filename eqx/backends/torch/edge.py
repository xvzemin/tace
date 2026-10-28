"""PyTorch evaluation of edge expressions."""

import torch

from ...utils.metadata import parse_metadata


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
