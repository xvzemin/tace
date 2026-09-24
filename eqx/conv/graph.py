################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Reusable source- and receiver-ordered convolution tiles."""

from collections import OrderedDict
from weakref import ref

import torch

_GRAPHS = OrderedDict()


def prepare_graph(source, target, owner):
    """Return an edge permutation grouped by its reduction node.

    Parameters
    ----------
    source, target : torch.Tensor
        Edge indices in the original order.
    owner : int
        Zero for source reductions, one for receiver reductions.

    Returns
    -------
    order : torch.Tensor
        Stable permutation grouping edges by their reduction node.

    Notes
    -----
    Views and detached aliases of unchanged index storage share cached plans.
    Kernels identify node boundaries inside bounded tiles of the sorted edges;
    no degree counts, prefix sums, or task buffers are needed. CUDA Graph
    capture rebuilds the permutation so replay remains correct when the input
    edge indices change. No tensor values are read by the host.
    """
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
                    root.untyped_storage().data_ptr(),
                    value.device,
                    value.dtype,
                    value.storage_offset(),
                    value.numel(),
                    value.stride(),
                    value._version,
                )
                for root, value in zip(roots, (source, target))
            ),
            owner,
        )
        if cacheable
        else None
    )
    cached = _GRAPHS.get(key)
    if cached is not None and all(saved() is not None for saved in cached[0]):
        _GRAPHS.move_to_end(key)
        if source.is_cuda:
            stream = torch.cuda.current_stream(source.device)
            if stream.cuda_stream != cached[3]:
                stream.wait_event(cached[2])
                cached[1].record_stream(stream)
        return cached[1]

    index = source if owner == 0 else target
    order = torch.argsort(index, stable=True)
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
            ready,
            stream_id,
        )
        if len(_GRAPHS) > 64:
            _GRAPHS.popitem(last=False)
    return order
