################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Reusable source- and receiver-ordered convolution tiles."""

from collections import OrderedDict
from weakref import ref

import torch

_GRAPHS = OrderedDict()


def prepare_graph(source, target, num_nodes, owner, row_size):
    """Return an edge permutation and bounded node-reduction tasks.

    Parameters
    ----------
    source, target : torch.Tensor
        Edge indices in the original order.
    num_nodes : int
        Number of nodes along the reduction axis.
    owner : int
        Zero for source reductions, one for receiver reductions.
    row_size : int
        Maximum number of edges assigned to one task.

    Returns
    -------
    order : torch.Tensor
        Stable permutation grouping edges by their reduction node.
    tasks : torch.Tensor
        Rows containing node index, sorted-edge start and stop, and whether
        the task owns the complete row. Empty tasks have equal start/stop.

    Notes
    -----
    Views of the same unmodified topology share cached plans. CUDA Graph
    capture rebuilds the indices with fixed-shape tensor operations so replay
    remains correct when the input edge indices change. No tensor values are
    read by the host.
    """
    if not num_nodes:
        return source.new_empty(0), source.new_empty((0, 4))
    roots = tuple(
        value._base if value._base is not None else value for value in (source, target)
    )
    capturing = source.is_cuda and torch.cuda.is_current_stream_capturing()
    cacheable = not capturing and not any(
        value.is_inference() for value in (source, target)
    )
    key = (
        (
            tuple(
                (
                    id(root),
                    value.storage_offset(),
                    value.numel(),
                    value.stride(),
                    value._version,
                )
                for root, value in zip(roots, (source, target))
            ),
            num_nodes,
            owner,
            row_size,
        )
        if cacheable
        else None
    )
    cached = _GRAPHS.get(key)
    if cached is not None and all(
        saved() is root for saved, root in zip(cached[0], roots)
    ):
        _GRAPHS.move_to_end(key)
        if source.is_cuda:
            stream = torch.cuda.current_stream(source.device)
            if stream.cuda_stream != cached[4]:
                stream.wait_event(cached[3])
                cached[1].record_stream(stream)
                cached[2].record_stream(stream)
        return cached[1:3]

    index = source if owner == 0 else target
    order = torch.argsort(index, stable=True)
    counts = index.new_zeros(num_nodes).scatter_add_(0, index, torch.ones_like(index))
    row_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    tiles = torch.div(counts + row_size - 1, row_size, rounding_mode="floor")
    tile_ptr = torch.cat((tiles.new_zeros(1), tiles.cumsum(0)))
    # This upper bound avoids data-dependent output shapes and host syncs.
    task = torch.arange(num_nodes + index.numel() // row_size, device=index.device)
    nodes = torch.searchsorted(tile_ptr[1:], task, right=True)
    valid = nodes < num_nodes
    nodes = nodes.clamp_max(num_nodes - 1)
    start = row_ptr[nodes] + (task - tile_ptr[nodes]) * row_size
    stop = torch.minimum(start + row_size, row_ptr[nodes + 1])
    start = torch.where(valid, start, 0)
    stop = torch.where(valid, stop, 0)
    exclusive = counts[nodes] <= row_size
    tasks = torch.stack((nodes, start, stop, exclusive), dim=1).contiguous()
    if cacheable:
        ready, stream_id = None, None
        if source.is_cuda:
            stream = torch.cuda.current_stream(source.device)
            ready = torch.cuda.Event()
            ready.record(stream)
            stream_id = stream.cuda_stream
        _GRAPHS[key] = (
            tuple(
                ref(root, lambda _, key=key: _GRAPHS.pop(key, None)) for root in roots
            ),
            order,
            tasks,
            ready,
            stream_id,
        )
        if len(_GRAPHS) > 64:
            _GRAPHS.popitem(last=False)
    return order, tasks
