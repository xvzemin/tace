"""Edge ordering and graph reductions."""

from collections import OrderedDict
from functools import lru_cache
from weakref import ref

import torch

_GRAPHS = OrderedDict()


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
            num_nodes,
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
                for value in (
                    cached[1] if isinstance(cached[1], tuple) else (cached[1],)
                ):
                    value.record_stream(stream)
        return cached[1]

    index = source if owner == 0 else target
    if num_nodes is None:
        result = torch.argsort(index, stable=True)
    else:
        order = prepare_graph(source, target, owner)
        boundaries = torch.arange(num_nodes + 1, device=index.device, dtype=index.dtype)
        pointers = torch.searchsorted(index[order], boundaries)
        result = order, pointers
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
            result,
            ready,
            stream_id,
        )
        if len(_GRAPHS) > 64:
            _GRAPHS.popitem(last=False)
    return result


@lru_cache(maxsize=64)
def reduction_source(dtype, width, channels, heads):
    """Generate a weighted graph-reduction kernel."""
    scalar = "double" if dtype == torch.float64 else "float"
    weight = (
        f" * weights[edge * {heads} + (column % {channels}) / {channels // heads}]"
        if heads
        else ""
    )
    return f"""
using T = {scalar};
using int64_t = long long;
extern "C" __global__ void run(const T* values, const T* weights,
    const int64_t* index, const int64_t* order, T* output, int64_t edges) {{
    const int64_t node = blockIdx.x;
    __shared__ int64_t bounds[2];
    if (threadIdx.x < 2) {{
        int64_t lo = 0, hi = edges;
        while (lo < hi) {{
            const int64_t mid = (lo + hi) / 2;
            if (index[order[mid]] < node + threadIdx.x) lo = mid + 1;
            else hi = mid;
        }}
        bounds[threadIdx.x] = lo;
    }}
    __syncthreads();
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (column >= {width} || bounds[0] == bounds[1]) return;
    T value = 0;
    for (int64_t k = bounds[0]; k < bounds[1]; ++k) {{
        const int64_t edge = order[k];
        value += values[edge * {width} + column]{weight};
    }}
    output[node * {width} + column] += value;
}}
"""


def reduce_messages(values, target, output, weights=None, channels=None):
    """Add weighted edge features to target nodes in place.

    CUDA inputs are contiguous. Derivatives are supplied by the enclosing
    convolution operator.
    """
    from ..kernels.cuda import kernels, runtime

    if not target.numel() or not output.numel():
        return
    values = values.contiguous()
    width = values.shape[1]
    heads = 0 if weights is None else weights.shape[1]
    channels = width if channels is None else channels
    order = prepare_graph(target, target, 1)
    code = reduction_source(values.dtype, width, channels, heads)
    kernel = kernels([code], values.device)[code]
    runtime().launch(
        [
            (
                kernel,
                [
                    values.data_ptr(),
                    0 if weights is None else weights.data_ptr(),
                    target.data_ptr(),
                    order.data_ptr(),
                    output.data_ptr(),
                    target.numel(),
                ],
                output.shape[0],
                (width + 127) // 128,
                128,
                0,
            )
        ],
        torch.cuda.current_stream(values.device).cuda_stream,
    )
