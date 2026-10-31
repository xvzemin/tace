"""Channel contractions on separate irreducible blocks."""

import math

import torch


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
