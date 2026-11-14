"""Channel contractions on separate irreducible blocks."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ....utils.metadata import parse_metadata
from ....utils.outputs import linear_output
from ... import config
from .grouped import element_expert_path, element_grouping, element_path


class SharedCoefficients(torch.autograd.Function):
    """Add independent coefficient parameters without aliasing their gradients."""

    generate_vmap_rule = True

    @staticmethod
    def forward(weight, shared):
        return weight + shared

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.weight_shape, ctx.shared_shape = (value.shape for value in inputs)

    @staticmethod
    def backward(ctx, gradient):
        return (
            gradient.sum_to_size(ctx.weight_shape),
            gradient.sum_to_size(ctx.shared_shape).clone(),
        )

    @staticmethod
    def jvp(ctx, weight_tangent, shared_tangent):
        return weight_tangent + shared_tangent


def indexed_parameters(weight, node_type, target_type, experts, numel):
    """Flatten independent element axes without copying the weight storage."""
    axes = 1 if target_type is None else 2
    if target_type is not None:
        node_type = node_type * weight.shape[1] + target_type
    return weight.reshape(math.prod(weight.shape[:axes]), experts, numel), node_type


def contract(metadata, role, node_type, features, weight, output):
    """Evaluate indexed maps with bounded packing for large transposed programs."""
    paths = parse_metadata(metadata)[4]
    if (
        not config.TORCH_REFERENCE
        and paths
        and grouped_linear_available(features, weight, paths)
        and all(features.shape[0] * ci * co >= 2**25 for _, _, _, ci, co, _, _ in paths)
    ):
        return contract_grouped(metadata, role, node_type, features, weight, output)
    return contract_reference(metadata, role, node_type, features, weight, output)


def contract_reference(metadata, role, node_type, features, weight, output):
    """Evaluate indexed linear maps and either transpose with ordinary Torch."""
    din, dout, numel, experts, paths, layout = parse_metadata(metadata)
    if role == 2:
        return element_linear_reference(
            features, weight, node_type, paths, dout, layout
        )
    zero = features[..., :0].sum() + weight[..., :0].sum() + output[..., :0].sum()
    result = (
        torch.zeros_like(
            linear_output(metadata, role, node_type, features, weight, output)
        )
        + zero
    )
    nodes = node_type.shape[0]
    for sx, sw, sy, ci, co, dim, scale in paths:
        y = output[:, sy : sy + experts * co * dim]
        if layout == "mul_ir":
            y = y.reshape(nodes, experts, co, dim).transpose(-1, -2)
        else:
            y = y.reshape(nodes, dim, experts, co).transpose(1, 2)
        if role == 0:
            matrix = weight[:, :, sw : sw + ci * co].reshape(-1, experts, ci, co)
            value = (y @ matrix[node_type].transpose(-1, -2)) * scale
            value = (
                value.transpose(-1, -2) if layout == "mul_ir" else value.transpose(1, 2)
            )
            indices = sx + torch.arange(experts * ci * dim, device=node_type.device)
            result = result.index_add(1, indices, value.flatten(1))
        else:
            x = features[:, sx : sx + experts * ci * dim]
            if layout == "mul_ir":
                x = x.reshape(nodes, experts, ci, dim)
            else:
                x = x.reshape(nodes, dim, experts, ci).permute(0, 2, 3, 1)
            value = ((x @ y) * scale).flatten(2)
            value = value.new_zeros((weight.shape[0], experts, ci * co)).index_add(
                0, node_type, value
            )
            indices = sw + torch.arange(ci * co, device=node_type.device)
            result = result.index_add(2, indices, value)
    return result


def contract_grouped(metadata, role, node_type, features, weight, output):
    """Apply either transpose using one coefficient matrix per packed block."""
    din, dout, numel, experts, paths, layout = parse_metadata(metadata)
    if role == 2:
        return element_linear_grouped(features, weight, node_type, paths, dout, layout)
    zero = features[..., :0].sum() + weight[..., :0].sum() + output[..., :0].sum()
    result = (
        torch.zeros_like(
            linear_output(metadata, role, node_type, features, weight, output)
        )
        + zero
    )
    nodes, elements = node_type.shape[0], weight.shape[0]
    width = 128 if nodes >= 64 * elements else 32 if nodes >= 16 * elements else 8
    grouping = element_grouping(node_type, elements, experts, width)
    slots, owners, chunks = grouping

    def pack(value, start, channels, dim):
        value = value[:, start : start + experts * dim * channels]
        if layout == "mul_ir":
            value = value.reshape(nodes, experts, channels, dim).transpose(-1, -2)
        else:
            value = value.reshape(nodes, dim, experts, channels).transpose(1, 2)
        value = value.reshape(nodes * experts, dim, channels)
        return (
            value.new_zeros((chunks * width, dim, channels))
            .index_copy(0, slots, value)
            .reshape(chunks, width * dim, channels)
        )

    for sx, sw, sy, ci, co, dim, scale in paths:
        if role == 0:
            matrix = weight[..., sw : sw + ci * co].reshape(elements, experts, ci, co)
            matrix = matrix.transpose(-1, -2).flatten(-2)
            value = element_path(
                output, matrix, grouping, (sy, 0, 0, co, ci, dim, scale), layout, width
            )
            indices = sx + torch.arange(experts * ci * dim, device=node_type.device)
            result = result.index_add(1, indices, value)
        else:
            x, y = pack(features, sx, ci, dim), pack(output, sy, co, dim)
            value = x.transpose(1, 2) @ y
            value = value.new_zeros((elements * experts, ci, co)).index_add(
                0, owners, value
            )
            value = value.reshape(elements, experts, ci * co) * scale
            indices = sw + torch.arange(ci * co, device=node_type.device)
            result = result.index_add(2, indices, value)
    return result


def grouped_linear_available(features, weight, paths, *, dynamic=False):
    """Use bounded packing when it avoids large per-node weight tensors."""
    if not config.TORCH_GROUPED_LINEAR or not paths:
        return False
    nodes = features.shape[0]
    if isinstance(nodes, torch.SymInt):
        return (
            dynamic
            and features.device.type == "cpu"
            and all(
                statically_known_true(ci * co >= 4096)
                for _, _, _, ci, co, _, _ in paths
            )
            and statically_known_true(nodes >= 256)
            and statically_known_true(nodes >= 4 * weight.shape[0])
        )
    if features.device.type == "cpu":
        if not all(ci * co >= 4096 for _, _, _, ci, co, _, _ in paths):
            return False
        return nodes >= 256 and nodes >= 4 * weight.shape[0]
    return (
        features.is_cuda
        and features.shape[0] >= 512
        and features.shape[0] >= 4 * weight.shape[0]
        and all(
            ci * co >= 4096 and features.shape[0] * ci * co >= 2**23
            for _, _, _, ci, co, _, _ in paths
        )
    )


def element_linear_reference(features, weight, node_type, paths, output_dim, layout):
    """Ordinary indexed matrices and batched products."""
    if not features.shape[0] or not paths:
        zero = features[..., :0].sum() + weight[..., :0].sum()
        return features.new_zeros((features.shape[0], output_dim)) + zero
    return element_linear_blocks(
        features.unsqueeze(1),
        weight[node_type],
        paths,
        output_dim,
        layout,
    ).squeeze(1)


def element_linear_grouped(features, weight, node_type, paths, output_dim, layout):
    """Use bounded GEMM blocks instead of a weight matrix for every node."""
    nodes = features.shape[0]
    zero = features[..., :0].sum() + weight[..., :0].sum()
    if not nodes or not paths:
        return features.new_zeros((nodes, output_dim)) + zero
    elements, experts = weight.shape[:2]
    width = (
        32
        if isinstance(nodes, torch.SymInt)
        else 128
        if statically_known_true(nodes >= 64 * elements)
        else 32
        if statically_known_true(nodes >= 16 * elements)
        else 8
        if statically_known_true(nodes < 16 * elements)
        else 32
    )
    tight_groups = (
        config.TORCH_GROUPED_LINEAR
        and not config.TORCH_REFERENCE
        and features.device.type == "cpu"
        and features.dtype == torch.float32
        and not isinstance(nodes, torch.SymInt)
        and layout == "mul_ir"
        and any(statically_known_true(experts == size) for size in (1, 2, 4, 8))
        and any(statically_known_true(width == size) for size in (8, 32))
        and (
            statically_known_true(experts == 1)
            or all(statically_known_true(ci == 128) for _, _, _, ci, _, _, _ in paths)
        )
        and all(
            statically_known_true(ci == co)
            and any(statically_known_true(ci == size) for size in (64, 128))
            and any(statically_known_true(dim == size) for size in (1, 3, 5))
            for _, _, _, ci, co, dim, _ in paths
        )
        and statically_known_true(nodes >= 256)
        and statically_known_true(nodes >= 4 * elements)
    )
    expert_groups = (
        features.device.type == "cpu"
        and not isinstance(nodes, torch.SymInt)
        and any(statically_known_true(experts == size) for size in (2, 4, 8))
        and statically_known_true(width == 32)
        and all(
            statically_known_true(ci == co)
            and any(statically_known_true(ci == size) for size in (64, 128))
            for _, _, _, ci, co, _, _ in paths
        )
        and statically_known_true(nodes >= 256)
        and (
            tight_groups
            or statically_known_true(
                experts * ((nodes + width - 1) // width)
                == (nodes * experts + width - 1) // width
            )
        )
    )
    grouping = element_grouping(
        node_type, elements, 1 if expert_groups else experts, width, tight=tight_groups
    )
    execute = element_expert_path if expert_groups else element_path
    blocks = {}
    for path in paths:
        sy = path[2]
        value = execute(features, weight, grouping, path, layout, width)
        blocks[sy] = value if sy not in blocks else blocks[sy] + value
    pieces, offset = [], 0
    for sy in sorted(blocks):
        value = blocks[sy]
        if sy > offset:
            pieces.append(features.new_zeros((nodes, sy - offset)) + zero)
        pieces.append(value)
        offset = sy + value.shape[-1]
    if offset < output_dim:
        pieces.append(features.new_zeros((nodes, output_dim - offset)) + zero)
    return (
        torch.cat(pieces, dim=-1) if pieces else features.new_zeros((nodes, 0)) + zero
    )


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
    for sy in sorted(blocks):
        value = blocks[sy]
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
    if expand_weights or config.TORCH_REFERENCE:
        return element_linear_reference(
            features, weight, node_type, paths, output_dim, layout
        )
    # Eager grouping can inspect element counts; captured programs cannot.
    if (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    ):
        execute = (
            element_linear_grouped
            if grouped_linear_available(features, weight, paths, dynamic=True)
            else element_linear_reference
        )
        return execute(features, weight, node_type, paths, output_dim, layout)
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


def linear_reference(inputs, weight, bias, layout, shape):
    """Evaluate each declared path independently with ordinary Torch AD."""
    input_shapes, output_shapes, paths, biases, groups = layout
    zero = weight[..., :0].sum() + bias[..., :0].sum()
    for value in inputs:
        zero = zero + value[..., :0].sum()
    outputs = [None] * len(output_shapes)
    for i, j, start, scale in paths:
        ci, co = input_shapes[i][1], output_shapes[j][1]
        matrix = weight[..., start : start + ci * co].reshape(
            *weight.shape[:-1], ci, co
        )
        value = torch.einsum("...di,...io->...do", inputs[i], matrix) * scale
        outputs[j] = value if outputs[j] is None else outputs[j] + value
    for j, dims in enumerate(output_shapes):
        if outputs[j] is None:
            outputs[j] = weight.new_zeros((*shape, *dims)) + zero
    for j, start, size in biases:
        outputs[j] = outputs[j] + bias[..., start : start + size].unsqueeze(-2)
    return [value.expand(*shape, *dims) for value, dims in zip(outputs, output_shapes)]


def linear(inputs, weight, bias, layout, shape):
    """Contract (..., dimension, channels) blocks with flat path weights."""
    if config.TORCH_REFERENCE:
        return linear_reference(inputs, weight, bias, layout, shape)
    input_shapes, output_shapes, paths, biases, groups = layout
    sizes = tuple(input_shapes[i][1] * output_shapes[j][1] for i, j, _, _ in paths)
    uniform_scale = bool(paths) and all(path[3] == paths[0][3] for path in paths)
    if uniform_scale:
        # Shared path normalization needs only one elementwise launch, while
        # each path keeps its own slice of the independent weight tensor.
        weight = weight * paths[0][3]
    matrices = [
        value.reshape(*weight.shape[:-1], input_shapes[i][1], output_shapes[j][1])
        for value, (i, j, _, _) in zip(weight.split(sizes, -1), paths)
    ]
    if not uniform_scale:
        matrices = [value * path[3] for value, path in zip(matrices, paths)]
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


def cartesian_linear_reference(
    features, weight, bias, output_shapes, groups, input_slices, group_sections
):
    """Evaluate Cartesian paths without grouping their channel GEMMs."""
    shape = torch.broadcast_shapes(
        features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
    )
    zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
    outputs = []
    for (mul, dim), group in zip(output_shapes, groups):
        values, biases = [], []
        for ins, section in group:
            if ins.i_in < 0:
                biases.append(bias[..., section].unsqueeze(-1))
            else:
                value = features[..., input_slices[ins.i_in]].reshape(
                    *features.shape[:-1], ins.path_shape[0], dim
                )
                matrix = weight[..., section].reshape(
                    *weight.shape[:-1], *ins.path_shape
                )
                values.append(
                    torch.einsum("...ui,...uv->...vi", value, matrix) * ins.path_weight
                )
        value = (
            sum(values[1:], values[0])
            if values
            else features.new_zeros((*shape, mul, dim)) + zero
        )
        for item in biases:
            value = value + item
        outputs.append(value.expand(*shape, mul, dim).flatten(-2))
    return (
        torch.cat(outputs, -1) if outputs else features.new_zeros((*shape, 0)) + zero
    ), zero


def cartesian_linear(
    features, weight, bias, output_shapes, groups, input_slices, group_sections
):
    """Mix Cartesian channels and retain gradient connections to unused operands."""
    if config.TORCH_REFERENCE:
        return cartesian_linear_reference(
            features, weight, bias, output_shapes, groups, input_slices, group_sections
        )
    shape = torch.broadcast_shapes(
        features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
    )
    zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
    outputs = []
    for (mul, dim), group, input_section in zip(output_shapes, groups, group_sections):
        inputs, weights, biases = [], [], []
        scales = [ins.path_weight for ins, _ in group if ins.i_in >= 0]
        postscale = (
            config.TORCH_LINEAR_POSTSCALE
            and weight.ndim == 1
            and bool(scales)
            and all(value == scales[0] for value in scales)
            and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
        )
        for ins, section in group:
            if ins.i_in < 0:
                biases.append(bias[..., section].unsqueeze(-1))
                continue
            if input_section is None:
                inputs.append(
                    features[..., input_slices[ins.i_in]].reshape(
                        *features.shape[:-1], ins.path_shape[0], dim
                    )
                )
            matrix = weight[..., section].reshape(*weight.shape[:-1], *ins.path_shape)
            weights.append(matrix if postscale else matrix * ins.path_weight)
        if weights:
            if input_section is not None:
                width = (input_section.stop - input_section.start) // dim
                x = features[..., input_section].reshape(
                    *features.shape[:-1], width, dim
                )
            else:
                x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-2)
            w = weights[0] if len(weights) == 1 else torch.cat(weights, dim=-2)
            value = torch.einsum("...ui,...uv->...vi", x, w)
            if postscale:
                value = value * scales[0]
            value = value.expand(*shape, mul, dim)
        else:
            value = features.new_zeros((*shape, mul, dim)) + zero
        for b in biases:
            value = value + b
        outputs.append(value)
    output = (
        torch.cat([x.flatten(-2) for x in outputs], -1)
        if outputs
        else features.new_zeros((*shape, 0)) + zero
    )
    return output, zero


def linear_flat(features, weight, bias, description):
    """Keep channel contractions in batched GEMM, including parameter adjoints."""
    inputs = description[0]
    values = [
        value.reshape(features.shape[0], dim, mul)
        for value, (dim, mul) in zip(
            features.split([dim * mul for dim, mul in inputs], -1), inputs
        )
    ]
    result = linear(values, weight, bias, description, features.shape[:-1])
    return (
        torch.cat([value.flatten(-2) for value in result], -1)
        if result
        else features[..., :0]
    )
