"""Bounded element groups expressed as ordinary differentiable Torch operations."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true


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
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    counts = ends - starts
    chunk_ends = torch.div(counts + width - 1, width, rounding_mode="floor").cumsum(0)
    chunk_starts = torch.cat((chunk_ends.new_zeros(1), chunk_ends[:-1]))
    ranks = positions - starts[sorted_groups]
    slots = (chunk_starts[sorted_groups] + ranks // width) * width + ranks % width
    slots = torch.empty_like(slots).index_copy(0, order, slots)
    nodes = node_type.shape[0]
    chunks = (
        experts * torch.sym_min(nodes, (nodes + (width - 1) * elements) // width)
        if tight
        else elements * experts + (nodes * experts + width - 1) // width
    )
    chunk_types = torch.searchsorted(
        chunk_ends, torch.arange(chunks, device=node_type.device), right=True
    ).clamp_max(elements * experts - 1)
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
    if (
        torch.compiler.is_compiling()
        or (features.device.type == "cpu" and get_proxy_mode() is not None)
    ) and not (
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
