"""Channel contractions on separate irreducible blocks."""

import math

import torch


def element_linear_blocks(features, weight, paths, output_dim, layout):
    """Contract batches of equal-length element groups."""
    groups, nodes = features.shape[:2]
    experts = weight.shape[1]
    zero = features[..., :0].sum() + weight[..., :0].sum()
    blocks = {}
    for sx, sw, sy, ci, co, dim, scale in paths:
        value = features[..., sx : sx + experts * ci * dim]
        if layout == "mul_ir":
            value = value.reshape(groups, nodes, experts, ci, dim).permute(
                0, 2, 1, 4, 3
            )
        else:
            value = value.reshape(groups, nodes, dim, experts, ci).permute(
                0, 3, 1, 2, 4
            )
        matrix = weight[..., sw : sw + ci * co].reshape(groups * experts, ci, co)
        value = (value.reshape(groups * experts, nodes * dim, ci) @ matrix) * scale
        value = value.reshape(groups, experts, nodes, dim, co)
        value = (
            value.permute(0, 2, 1, 4, 3)
            if layout == "mul_ir"
            else value.permute(0, 2, 3, 1, 4)
        )
        value = value.reshape(groups, nodes, experts * co * dim)
        blocks[sy] = value if sy not in blocks else blocks[sy] + value
    pieces, offset = [], 0
    for sy, value in sorted(blocks.items()):
        if sy > offset:
            pieces.append(features.new_zeros((groups, nodes, sy - offset)) + zero)
        pieces.append(value)
        offset = sy + value.shape[-1]
    if offset < output_dim:
        pieces.append(features.new_zeros((groups, nodes, output_dim - offset)) + zero)
    return torch.cat(pieces, dim=-1)


def element_linear(
    features, weight, node_type, paths, output_dim, layout, expand_weights=False
):
    """Apply indexed element matrices to irreducible features."""
    nodes = features.shape[0]
    zero = features[..., :0].sum() + weight[..., :0].sum()
    if not nodes or not paths:
        return features.new_zeros((nodes, output_dim)) + zero
    if expand_weights:
        return element_linear_blocks(
            features.unsqueeze(1),
            weight.index_select(0, node_type),
            paths,
            output_dim,
            layout,
        ).squeeze(1)
    types, inverse, counts = torch.unique(
        node_type, sorted=True, return_inverse=True, return_counts=True
    )
    sizes = counts.tolist()
    width = max(sizes)
    order = node_type.argsort()
    matrices = weight.index_select(0, types)
    if len(sizes) * width <= 2 * nodes:
        starts = counts.cumsum(0) - counts
        ranks = torch.arange(nodes, device=node_type.device) - starts[inverse[order]]
        ranks = torch.empty_like(order).index_copy(0, order, ranks)
        index = inverse * width + ranks
        packed = features.new_zeros((len(sizes) * width, features.shape[-1]))
        packed = packed.index_copy(0, index, features).reshape(len(sizes), width, -1)
        values = element_linear_blocks(packed, matrices, paths, output_dim, layout)
        return values.reshape(len(sizes) * width, output_dim).index_select(0, index)
    outputs = [
        element_linear_blocks(
            x.unsqueeze(0), w.unsqueeze(0), paths, output_dim, layout
        ).squeeze(0)
        for x, w in zip(
            features.index_select(0, order).split(sizes), matrices.unbind(0)
        )
    ]
    result = torch.cat(outputs, dim=0)
    return torch.empty_like(result).index_copy(0, order, result)


def linear_layout(inputs, outputs, instructions):
    """Describe channel matrices grouped by their input entries."""
    paths, biases = [], []
    offset = bias_offset = 0
    for ins in instructions:
        size = math.prod(ins.path_shape)
        if ins.i_in == -1:
            biases.append((ins.i_out, bias_offset, size))
            bias_offset += size
        else:
            paths.append((ins.i_in, ins.i_out, offset, ins.path_weight))
            offset += size
    groups = {}
    for j, (dim, _) in enumerate(outputs):
        selected = tuple(k for k, path in enumerate(paths) if path[1] == j)
        sources = tuple(paths[k][0] for k in selected)
        key = (dim, sources, j if len(set(sources)) != len(sources) else -1)
        groups.setdefault(key, []).append((j, selected))
    return (
        tuple(inputs),
        tuple(outputs),
        tuple(paths),
        tuple(biases),
        tuple(tuple(entries) for entries in groups.values()),
    )


def linear(inputs, weight, bias, layout, shape):
    """Contract (..., dimension, channels) blocks with flat path weights."""
    input_shapes, output_shapes, paths, biases, groups = layout
    sizes = tuple(input_shapes[i][1] * output_shapes[j][1] for i, j, _, _ in paths)
    matrices = [
        value.reshape(*weight.shape[:-1], input_shapes[i][1], output_shapes[j][1])
        * scale
        for value, (i, j, _, scale) in zip(weight.split(sizes, -1), paths)
    ]
    outputs = [None] * len(output_shapes)
    shared = all(size == 1 for size in weight.shape[:-1])
    for entries in groups:
        indices = entries[0][1]
        if not indices:
            zero = weight[..., :0].sum() + bias[..., :0].sum()
            for value in inputs:
                zero = zero + value[..., :0].sum()
            for j, _ in entries:
                outputs[j] = weight.new_zeros((*shape, *output_shapes[j])) + zero
            continue
        sources = [inputs[paths[k][0]] for k in indices]
        value = sources[0] if len(sources) == 1 else torch.cat(sources, -1)
        rows = []
        for row in range(len(indices)):
            parts = [matrices[selected[row]] for _, selected in entries]
            rows.append(parts[0] if len(parts) == 1 else torch.cat(parts, -1))
        matrix = rows[0] if len(rows) == 1 else torch.cat(rows, -2)
        if shared:
            result = (
                value.reshape(math.prod(value.shape[:-1]), value.shape[-1])
                @ matrix.reshape(matrix.shape[-2:])
            ).reshape(*value.shape[:-1], matrix.shape[-1])
        else:
            result = value @ matrix
        parts = (
            (result,)
            if len(entries) == 1
            else result.split([output_shapes[j][1] for j, _ in entries], -1)
        )
        for (j, _), part in zip(entries, parts):
            outputs[j] = part
    for j, start, size in biases:
        outputs[j] = outputs[j] + bias[..., start : start + size].unsqueeze(-2)
    return [value.expand(*shape, *dims) for value, dims in zip(outputs, output_shapes)]
