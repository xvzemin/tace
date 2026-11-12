"""Bounded element groups expressed as ordinary differentiable Torch operations."""

import torch


def element_grouping(node_type, elements, experts, width):
    """Pack rows in blocks of ``width`` without data-dependent allocations.

    The number of nonempty blocks is at most ``elements * experts +
    ceil(samples * experts / width)``. Unused blocks receive zero features.
    """
    types = torch.arange(elements, device=node_type.device).index_select(0, node_type)
    groups = (
        types[:, None] * experts + torch.arange(experts, device=node_type.device)
    ).flatten()
    sorted_groups, order = groups.sort()
    ends = torch.searchsorted(
        sorted_groups, torch.arange(1, elements * experts + 1, device=node_type.device)
    )
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    counts = ends - starts
    chunk_ends = torch.div(counts + width - 1, width, rounding_mode="floor").cumsum(0)
    chunk_starts = torch.cat((chunk_ends.new_zeros(1), chunk_ends[:-1]))
    ranks = torch.arange(
        groups.shape[0], device=node_type.device
    ) - starts.index_select(0, sorted_groups)
    slots = (
        chunk_starts.index_select(0, sorted_groups) + ranks // width
    ) * width + ranks % width
    slots = torch.empty_like(slots).index_copy(0, order, slots)
    chunks = elements * experts + (groups.shape[0] + width - 1) // width
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
    matrix = (
        weight[..., sw : sw + ci * co].reshape(-1, ci, co).index_select(0, chunk_types)
    )
    value = (packed.reshape(chunks, width * dim, ci) @ matrix).reshape(
        chunks * width, dim, co
    )
    value = value.index_select(0, slots).reshape(nodes, experts, dim, co)
    value = value.transpose(-2, -1) if layout == "mul_ir" else value.transpose(1, 2)
    return value.reshape(nodes, experts * co * dim) * scale
