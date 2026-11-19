"""Bounded element groups expressed as ordinary differentiable Torch operations."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ....utils.autograd import forward_ad_active, torch_gradient_mask
from ... import config


def element_grouping(node_type, elements, experts, width, *, tight=False):
    """Pack rows in blocks of ``width`` without data-dependent allocations.

    The default capacity is ``elements * experts + ceil(samples * experts /
    width)``. ``tight=True`` uses the sharp bound for repeated expert counts.
    Unused blocks receive zero features.
    """
    types = torch.arange(elements, device=node_type.device)[node_type]
    positions = torch.arange(node_type.shape[0] * experts, device=node_type.device)
    groups = (
        types
        if experts == 1
        else types[positions // experts] * experts + positions % experts
    )
    sorted_groups, order = groups.sort()
    ends = torch.searchsorted(
        sorted_groups, torch.arange(1, elements * experts + 1, device=node_type.device)
    )
    nodes = node_type.shape[0]
    chunks = (
        experts * torch.sym_min(nodes, (nodes + (width - 1) * elements) // width)
        if tight
        else elements * experts + (nodes * experts + width - 1) // width
    )
    return tile_layout(sorted_groups, order, ends, positions, width, chunks)


def tile_layout(sorted_groups, order, ends, positions, width, chunks):
    """Return the row slots and table indices for a sorted partition."""
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    counts = ends - starts
    chunk_ends = torch.div(counts + width - 1, width, rounding_mode="floor").cumsum(0)
    chunk_starts = torch.cat((chunk_ends.new_zeros(1), chunk_ends[:-1]))
    slots = positions + (chunk_starts * width - starts)[sorted_groups]
    slots = torch.empty_like(slots).scatter_(0, order, slots)
    chunk_types = torch.searchsorted(
        chunk_ends, torch.arange(chunks, device=order.device), right=True
    ).clamp_max(ends.shape[0] - 1)
    return slots, chunk_types, chunks


def element_path(features, weight, grouping, path, layout, width):
    """Apply a channel map with one matrix per block of element/expert rows."""
    sx, sw, sy, ci, co, dim, scale = path
    nodes, experts = features.shape[0], weight.shape[1]
    slots, chunk_types, chunks = grouping
    value = features[:, sx : sx + experts * ci * dim]
    if layout == "mul_ir":
        value = value.reshape(nodes, experts, ci, dim).transpose(-2, -1)
    else:
        value = value.reshape(nodes, dim, experts, ci).transpose(1, 2)
    value = value.reshape(nodes * experts, dim, ci)
    packed = value.new_zeros((chunks * width, dim, ci)).index_copy(0, slots, value)
    matrix = weight[..., sw : sw + ci * co].reshape(weight.shape[0] * experts, ci, co)[
        chunk_types
    ]
    if (torch.compiler.is_compiling() or get_proxy_mode() is not None) and not (
        statically_known_true(chunks == 0)
        or (
            statically_known_true(chunks > 0)
            and statically_known_true((chunks * width) % chunks == 0)
            and statically_known_true((chunks * width * dim) % (chunks * width) == 0)
        )
    ):
        # Both buffers are freshly allocated, contiguous and have offset zero.
        # Explicit strides avoid guards for symbolic reshape divisibility.
        value = (
            torch.as_strided(
                packed,
                (chunks, width * dim, ci),
                (width * dim * ci, ci, 1),
                storage_offset=0,
            )
            @ matrix
        )
        value = torch.as_strided(
            value,
            (chunks * width, dim, co),
            (dim * co, co, 1),
            storage_offset=0,
        )
    else:
        value = (packed.reshape(chunks, width * dim, ci) @ matrix).reshape(
            chunks * width, dim, co
        )
    value = value[slots].reshape(nodes, experts, dim, co)
    value = value.transpose(-2, -1) if layout == "mul_ir" else value.transpose(1, 2)
    return value.reshape(nodes, experts * co * dim) * scale


def element_expert_path(features, weight, grouping, path, layout, width):
    """Keep expert batches separate while packing each node only once."""
    sx, sw, sy, ci, co, dim, scale = path
    nodes, experts = features.shape[0], weight.shape[1]
    slots, owners, chunks = grouping
    value = features[:, sx : sx + experts * ci * dim]
    if layout == "mul_ir":
        value = value.reshape(nodes, experts, ci, dim).permute(1, 0, 3, 2)
    else:
        value = value.reshape(nodes, dim, experts, ci).permute(2, 0, 1, 3)
    packed = value.new_zeros((experts, chunks * width, dim, ci)).index_copy(
        1, slots, value
    )
    matrix = weight[..., sw : sw + ci * co].reshape(weight.shape[0], experts, ci, co)
    matrix = matrix.permute(1, 0, 2, 3).index_select(1, owners)
    value = packed.reshape(experts * chunks, width * dim, ci) @ matrix.reshape(
        experts * chunks, ci, co
    )
    value = value.reshape(experts, chunks * width, dim, co).index_select(1, slots)
    value = (
        value.permute(1, 0, 3, 2) if layout == "mul_ir" else value.permute(1, 2, 0, 3)
    )
    return value.reshape(nodes, experts * co * dim) * scale


def channel_groups(node_type, elements):
    """Pack nodes of the same element in blocks shared by all output irreps."""
    width = config.ACE_ELEMENT_BLOCK
    nodes = node_type.shape[0]
    if elements >= 32:
        capture = (
            torch.compiler.is_compiling()
            or get_proxy_mode() is not None
            or is_fake(node_type)
        )
        if capture and config.TORCH_ACE_PACK_TABLES and node_type.device.type == "cuda":
            if statically_known_true(nodes <= 2 * elements):
                width = max(1, width // 4)
            elif statically_known_true(nodes <= 8 * elements):
                # Narrower blocks save angular padding during joint differentiation.
                width = max(
                    1, width // 2 if torch.is_grad_enabled() else 3 * width // 4
                )
        elif not capture and isinstance(nodes, int) and nodes <= 2 * elements:
            width = max(1, width // 2)
    slots, owners, chunks = element_grouping(node_type, elements, 1, width, tight=True)
    return channel_rows(width, slots, owners, chunks, nodes)


def channel_rows(width, slots, owners, chunks, nodes, experts=1):
    """Map padded matrix rows back to samples and flag padding in adjoints."""
    if experts != 1:
        expert = torch.arange(experts, device=slots.device)
        slots = (
            ((slots // width)[:, None] * experts + expert) * width
            + (slots % width)[:, None]
        ).flatten()
        owners = (owners[:, None] * experts + expert).flatten()
        nodes, chunks = nodes * experts, chunks * experts
    rows = slots.new_full((chunks * width,), nodes).scatter(
        0, slots, torch.arange(nodes, device=slots.device)
    )
    return width, rows.clamp_max(nodes - 1), slots, owners, rows != nodes


def channel_tiles(groups, matrix_bytes, nodes, elements):
    """Bound the coefficient scratch independently of the size of the element table."""
    count = groups.numel()
    budget = config.ACE_POLYNOMIAL_WORKSPACE
    if elements >= 32 and nodes <= 2 * elements:
        budget = min(budget, 4 << 20)
    step = max(1, budget // matrix_bytes)
    return (slice(i, min(i + step, count)) for i in range(0, count, step))


class IndexedChannelProduct(torch.autograd.Function):
    """Apply element matrices or their transposes without saving expanded weights."""

    @staticmethod
    def forward(ctx, transpose, width, rows, slots, owners, valid, features, *weights):
        ctx.transpose, ctx.width = transpose, width
        ctx.save_for_backward(rows, slots, owners, valid, features, *weights)
        inputs, outputs = sum(w.shape[1] for w in weights), weights[0].shape[2]
        angular = features.shape[1]
        out = features.new_empty(
            (rows.numel(), angular, inputs if transpose else outputs)
        )
        for section in channel_tiles(
            owners,
            inputs * outputs * features.element_size(),
            features.shape[0],
            weights[0].shape[0],
        ):
            group = owners[section]
            node = slice(section.start * width, section.stop * width)
            parts = [w.index_select(0, group) for w in weights]
            matrix = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            values = features.index_select(0, rows[node]).reshape(
                group.numel(), width * angular, features.shape[2]
            )
            torch.bmm(
                values,
                matrix.transpose(1, 2) if transpose else matrix,
                out=out[node].view(group.numel(), width * angular, out.shape[2]),
            )
            del matrix, parts
        return out.index_select(0, slots)

    @staticmethod
    def backward(ctx, gradient):
        rows, slots, owners, valid, x, *weights = ctx.saved_tensors
        grouping = ctx.width, rows, slots, owners, valid
        dx, *dw = torch_gradient_mask(ctx, ctx.needs_input_grad[6:])
        derivative = (
            IndexedChannelProduct.apply(
                not ctx.transpose, *grouping, gradient, *weights
            )
            if dx
            else None
        )
        coefficients = (
            IndexedChannelGradient.apply(
                tuple(w.shape[1] for w in weights),
                weights[0].shape[0],
                *grouping,
                *((gradient, x) if ctx.transpose else (x, gradient)),
            )
            if any(dw)
            else (None,) * len(weights)
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            derivative,
            *(g if want else None for g, want in zip(coefficients, dw)),
        )


class IndexedChannelGradient(torch.autograd.Function):
    """Reduce channel outer products directly into their original element tables."""

    @staticmethod
    def forward(ctx, sizes, elements, width, rows, slots, owners, valid, x, y):
        ctx.width = width
        ctx.save_for_backward(rows, slots, owners, valid, x, y)
        results = tuple(x.new_zeros((elements, size, y.shape[2])) for size in sizes)
        angular = x.shape[1]
        for section in channel_tiles(
            owners, x.shape[2] * y.shape[2] * x.element_size(), x.shape[0], elements
        ):
            group = owners[section]
            node = slice(section.start * width, section.stop * width)
            index, padding = rows[node], ~valid[node, None, None]
            a, b = (
                v.index_select(0, index)
                .masked_fill_(padding, 0)
                .reshape(group.numel(), width * angular, v.shape[2])
                for v in (x, y)
            )
            parts = (a.transpose(1, 2) @ b).split(sizes, dim=1)
            for result, part in zip(results, parts):
                result.index_add_(0, group, part)
        return results

    @staticmethod
    def backward(ctx, *weights):
        rows, slots, owners, valid, x, y = ctx.saved_tensors
        grouping = ctx.width, rows, slots, owners, valid
        dx, dy = torch_gradient_mask(ctx, ctx.needs_input_grad[-2:])
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            IndexedChannelProduct.apply(True, *grouping, y, *weights) if dx else None,
            IndexedChannelProduct.apply(False, *grouping, x, *weights) if dy else None,
        )


def indexed_channel_product(features, weights, grouping):
    """Contract a node's element tables without gathering a matrix per node."""
    if not (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or is_fake(features)
        or forward_ad_active()
    ):
        return IndexedChannelProduct.apply(False, *grouping, features, *weights)
    width, _, slots, owners, _ = grouping
    groups, dim, inputs = owners.shape[0], features.shape[1], features.shape[2]
    elements = weights[0].shape[0]
    if (
        config.TORCH_ACE_PACK_TABLES
        and features.device.type == "cuda"
        and elements >= 32
        and statically_known_true(features.shape[0] >= elements)
        and statically_known_true(features.shape[0] <= 8 * elements)
    ):
        # One lookup also gives autograd one scatter across all body orders.
        table = weights[0] if len(weights) == 1 else torch.cat(weights, dim=1)
        matrix = table.index_select(0, owners)
    else:
        parts = [w.index_select(0, owners) for w in weights]
        matrix = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    outputs = matrix.shape[2]
    packed = features.new_zeros((groups * width, dim, inputs)).index_copy(
        0, slots, features
    )
    packed = torch.as_strided(
        packed, (groups, width * dim, inputs), (width * dim * inputs, inputs, 1)
    )
    result = torch.bmm(packed, matrix)
    result = torch.as_strided(
        result, (groups * width, dim, outputs), (dim * outputs, outputs, 1)
    )
    return result.index_select(0, slots)
