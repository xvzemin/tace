"""Differentiable gathers, group packing and sparse column sums."""

import torch

from ....utils.metadata import parse_metadata


def gather_sum(features, indices):
    """Sum optionally indexed feature matrices without changing their row order."""
    if not features or len(features) != len(indices):
        raise ValueError("Each feature matrix requires an index tensor or None.")
    values = [
        value if index is None else value.index_select(0, index)
        for value, index in zip(features, indices)
    ]
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("Gathered feature matrices must have the same shape.")
    return sum(values[1:], values[0])


def grouped_permute(features, indices, inverse, widths, transpose=False):
    """Pack contiguous batch matrices by group, or undo the same permutation."""
    widths = parse_metadata(widths)
    if sum(widths) != features.shape[1]:
        raise ValueError("Group widths must sum to the feature width.")
    if not sum(widths):
        return features.clone()
    rows, start = features.shape[0], 0
    parts = []
    if transpose:
        flat = features.flatten()
        for width in widths:
            parts.append(
                flat[rows * start : rows * (start + width)].reshape(rows, width)
            )
            start += width
        return torch.cat(parts, dim=1).index_select(1, indices)
    values = features.index_select(1, indices)
    for width in widths:
        parts.append(values[:, start : start + width].flatten())
        start += width
    return torch.cat(parts).reshape(features.shape)


def indexed_sum(
    features, indices, ptr, transpose_indices, transpose_ptr, count, transpose=False
):
    """Apply a binary CSR column map and its declared normalization."""
    width = ptr.numel() - 1
    columns = torch.repeat_interleave(
        torch.arange(width, device=features.device),
        ptr[1:] - ptr[:-1],
        output_size=indices.numel(),
    )
    values = features.index_select(1, indices)
    if transpose and count.numel():
        values = values / count[indices]
    result = features.new_zeros((features.shape[0], width)).index_add(
        1, columns, values
    )
    return result / count if count.numel() and not transpose else result
