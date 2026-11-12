"""Reuse live graph plans within one TACE-OAM-XL-v2 program."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import ref

import torch

from equivariantx.ops.kernels.cuda.reduction import csr_source

_GRAPHS = OrderedDict()
_GRAPH_SCOPE = ContextVar("eqx_graph_plan_scope", default=None)


@contextmanager
def graph_plan_scope():
    """Reuse live graph sorting only within one complete program invocation.

    Captured sorts stay in the graph and execute on every replay. The scoped
    dictionary is discarded at exit; it never caches values across batches.
    Versioned tensor keys invalidate entries after an in-place index change.
    """
    if _GRAPH_SCOPE.get() is not None:
        yield
        return
    token = _GRAPH_SCOPE.set({})
    try:
        yield
    finally:
        _GRAPH_SCOPE.reset(token)



def prepare_graph(source, target, owner, num_nodes=None):
    """Return an edge permutation grouped by its reduction node.

    Parameters
    ----------
    source, target : torch.Tensor
        Source and target indices of shape ``(edges,)``.
    owner : int
        Zero for source reductions, one for target reductions.
    num_nodes : int, optional
        Also return CSR offsets, including empty nodes.

    Returns
    -------
    order : torch.Tensor
        Stable permutation grouping edges by their reduction node.
    pointers : torch.Tensor
        CSR offsets of shape ``(num_nodes + 1,)``, when requested.
    """
    index = source if owner == 0 else target
    root = index._base if index._base is not None else index
    capturing = index.is_cuda and torch.cuda.is_current_stream_capturing()
    scope = _GRAPH_SCOPE.get()
    scoped = scope is not None and not index.is_inference()
    cacheable = not capturing and not index.is_inference()
    key = (
        (
            root.untyped_storage().data_ptr(),
            index.device,
            index.dtype,
            index.storage_offset(),
            index.numel(),
            index.stride(),
            index._version,
            num_nodes,
        )
        if cacheable or scoped
        else None
    )
    if scoped and key in scope:
        return scope[key]
    cached = _GRAPHS.get(key) if cacheable else None
    if cached is not None and cached[0]() is not None:
        _GRAPHS.move_to_end(key)
        if index.is_cuda:
            stream = torch.cuda.current_stream(index.device)
            if stream.cuda_stream != cached[3]:
                stream.wait_event(cached[2])
                for value in (
                    cached[1] if isinstance(cached[1], tuple) else (cached[1],)
                ):
                    value.record_stream(stream)
        if scoped:
            scope[key] = cached[1]
        return cached[1]

    if num_nodes is None:
        result = torch.argsort(index, stable=True)
    else:
        order = prepare_graph(source, target, owner)
        if (
            index.is_cuda
            and index.dtype in (torch.int32, torch.int64)
            and num_nodes >= 0
        ):
            from equivariantx.ops.kernels.cuda.runtime import kernels, runtime

            pointers = order.new_empty(num_nodes + 1)
            code = csr_source(index.dtype)
            kernel = kernels([code], index.device)[code]
            runtime().launch(
                [
                    (
                        kernel,
                        [
                            index.data_ptr(),
                            order.data_ptr(),
                            pointers.data_ptr(),
                            index.numel(),
                            num_nodes,
                            index.stride(0),
                        ],
                        min((num_nodes + 256) // 256, 65535),
                        1,
                        256,
                        0,
                    )
                ],
                torch.cuda.current_stream(index.device).cuda_stream,
            )
        else:
            boundaries = torch.arange(
                num_nodes + 1, device=index.device, dtype=index.dtype
            )
            pointers = torch.searchsorted(index[order], boundaries)
        result = order, pointers
    if scoped:
        scope[key] = result
    if cacheable:
        ready, stream_id = None, None
        if index.is_cuda:
            stream = torch.cuda.current_stream(index.device)
            ready = torch.cuda.Event()
            ready.record(stream)
            stream_id = stream.cuda_stream
        _GRAPHS[key] = (
            ref(root, lambda _, key=key: _GRAPHS.pop(key, None)),
            result,
            ready,
            stream_id,
        )
        if len(_GRAPHS) > 64:
            _GRAPHS.popitem(last=False)
    return result
