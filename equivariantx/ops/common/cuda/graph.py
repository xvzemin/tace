"""Bounded CUDA Graph replay of native contractions and their adjoints."""

import os
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from threading import RLock

import torch

from ....utils.metadata import parse_metadata

ENABLED = {
    "o3": True,
    "o2_o3": True,
    "ace": True,
    "product": True,
    "edge_program": True,
}
EDGE_BUCKET = 2048
MIN_EDGES = 1024
ROW_BUCKET = 128
MIN_ROWS = 64
MAX_GRAPHS = 128
MAX_BYTES = 12 << 30
CONVOLUTION_GRAPH_MIN_CALLS = 3
CAPTURE_CONSTANTS = ContextVar("eqx_cuda_capture_constants", default=None)
_GRAPHS = OrderedDict()
_CANDIDATES = OrderedDict()
_LOCK = RLock()


@lru_cache(maxsize=64)
def capture_pool(device, stream):
    """Share scratch space between serialized replays with private outputs."""
    return torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=device)


@contextmanager
def retain_constants(constants):
    token = CAPTURE_CONSTANTS.set(constants)
    try:
        yield
    finally:
        CAPTURE_CONSTANTS.reset(token)


def enabled(name):
    """Return whether native graph replay is enabled for this operator."""
    default = "1" if name == "product" else "0"
    return os.environ.get("EQX_USE_CUDA_GRAPH", default) == "1" and ENABLED[name]


def execute(name, key, function, inputs, outputs, shapes=None, fills=None, read=None):
    """Capture a native backend, leaving its registered derivatives unchanged.

    Inputs and outputs are copied across the capture boundary so results never
    alias reusable buffers. Shape-only operands are not copied. Outer captures
    execute the original backend directly; they own their capture lifetime.
    The cache is bounded across models, shapes and derivative programs.
    """
    device = inputs[0].device
    if (
        not enabled(name)
        or not inputs[0].is_cuda
        or not any(x.numel() for x in outputs)
    ):
        function(inputs, outputs)
        return
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            function(inputs, outputs)
            return
    input_shapes, output_shapes = shapes or (
        tuple(x.shape for x in inputs),
        tuple(x.shape for x in outputs),
    )
    read = tuple(range(len(inputs))) if read is None else tuple(sorted(read))
    fills = {} if fills is None else fills
    stream = torch.cuda.current_stream(device)
    signature = (
        name,
        key,
        device,
        stream.cuda_stream,
        tuple((shape, x.dtype) for shape, x in zip(input_shapes, inputs)),
        tuple((shape, x.dtype) for shape, x in zip(output_shapes, outputs)),
        tuple(sorted(fills.items())),
        read,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    estimated = sum(
        torch.Size(shape).numel() * x.element_size()
        for shape, x in zip((*input_shapes, *output_shapes), (*inputs, *outputs))
    )
    capacity = min(
        MAX_BYTES, torch.cuda.get_device_properties(device).total_memory // 16
    )
    if estimated > capacity // 2:
        function(inputs, outputs)
        return
    if name == "edge_program":
        with _LOCK:
            defer = False
            if signature not in _GRAPHS:
                visits = _CANDIDATES.pop(signature, 0) + 1
                defer = visits < CONVOLUTION_GRAPH_MIN_CALLS
                if defer:
                    _CANDIDATES[signature] = visits
                    if len(_CANDIDATES) > MAX_GRAPHS:
                        _CANDIDATES.popitem(last=False)
        if defer:
            function(inputs, outputs)
            return
    with _LOCK, torch.cuda.device(device):
        cached = _GRAPHS.get(signature)
        if cached is None:
            before = torch.cuda.memory_allocated(device)
            reserved = (
                torch.cuda.memory_reserved(device) if name == "edge_program" else 0
            )
            values = [x.new_zeros(shape) for shape, x in zip(input_shapes, inputs)]
            results = [x.new_empty(shape) for shape, x in zip(output_shapes, outputs)]
        else:
            graph, values, results = cached[:3]
        for i in read:
            value, x = values[i], inputs[i]
            if value.shape != x.shape:
                value[x.shape[0] :].fill_(fills.get(i, 0))
                value = value[: x.shape[0]]
            value.copy_(x)
        if cached is None:
            if name in ("product", "edge_program"):
                pool, capture_stream = capture_pool(device, stream.cuda_stream)
            else:
                pool, capture_stream = None, torch.cuda.Stream(device=device)
            capture_stream.wait_stream(stream)
            constants = {}
            with retain_constants(constants), torch.cuda.stream(capture_stream):
                # Compile and initialize every native phase before capture.
                function(values, results)
                function(values, results)
            stream.wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            # Pooled captures retain no outputs allocated inside the graph:
            # all persistent inputs/results were allocated above and outputs
            # are copied out. Temporary storage can therefore be reused even
            # when shape buckets or derivative programs replay in a new order.
            # Loader threads can allocate unrelated pinned memory concurrently.
            with (
                retain_constants(constants),
                torch.cuda.graph(
                    graph,
                    stream=capture_stream,
                    pool=pool,
                    capture_error_mode="thread_local",
                ),
            ):
                function(values, results)
            stream.wait_stream(capture_stream)
            size = max(estimated, torch.cuda.memory_allocated(device) - before)
            if name == "edge_program":
                size = max(size, torch.cuda.memory_reserved(device) - reserved)
            _GRAPHS[signature] = graph, values, results, size, constants
            while len(_GRAPHS) > 1 and (
                len(_GRAPHS) > MAX_GRAPHS
                or sum(entry[3] for entry in _GRAPHS.values()) > capacity
            ):
                _GRAPHS.popitem(last=False)
        _GRAPHS.move_to_end(signature)
        graph.replay()
        for output, result in zip(outputs, results):
            output.copy_(result[: output.shape[0]])


def convolution(
    name, key, function, source, target, operands, outputs, program, output_role
):
    """Bucket edges using an isolated zero node, including transposed programs.

    Padded edges connect only the extra source and target nodes. Their outputs
    are discarded, and node cotangents at those nodes are zero. This preserves
    shared weights, unweighted paths and arbitrary derivatives without changing
    any CUDA contraction formula or dropping paths.
    """
    if not enabled(name) or not source.is_cuda or source.numel() < MIN_EDGES:
        function((source, target, *operands), outputs)
        return
    with torch.cuda.device(source.device):
        if torch.cuda.is_current_stream_capturing():
            function((source, target, *operands), outputs)
            return
    edges = source.numel()
    capacity = (edges + EDGE_BUCKET - 1) // EDGE_BUCKET * EDGE_BUCKET
    kinds = [None] * len(operands)
    read = {0, 1}
    destinations = {}
    for mapping, _, pairs in program:
        for role, index in enumerate(mapping):
            kind = (
                "node"
                if role in (0, output_role)
                else "fixed"
                if role == 2 or operands[index].shape[0] == 1
                else "edge"
            )
            if kinds[index] not in (None, kind):
                function((source, target, *operands), outputs)
                return
            kinds[index] = kind
        for output, slot in pairs:
            destinations[slot] = mapping[output]
            read.update(
                2 + index for role, index in enumerate(mapping) if role != output
            )
    mapping = program[0][0]
    nodes = operands[mapping[0]].shape[0], operands[mapping[output_role]].shape[0]
    shapes = tuple(
        (x.shape[0] + 1, *x.shape[1:])
        if kind == "node"
        else (capacity, *x.shape[1:])
        if kind == "edge"
        else x.shape
        for x, kind in zip(operands, kinds)
    )
    execute(
        name,
        key,
        function,
        (source, target, *operands),
        outputs,
        shapes=(
            ((capacity,), (capacity,), *shapes),
            tuple(shapes[destinations[i]] for i in range(len(outputs))),
        ),
        fills={0: nodes[0], 1: nodes[1]},
        read=read,
    )


@lru_cache(maxsize=128)
def is_convolution(metadata):
    operations = {node[0] for node in parse_metadata(metadata)[0]}
    return "convolution" in operations or {"frame", "matmul"} <= operations


def padded_rows(rows, minimum):
    step = max(minimum, 1 << max(0, rows.bit_length() - 4))
    return (rows + step - 1) // step * step


def indexed_replay(metadata, function, values, outputs, layout, counts):
    """Direct padded edges to private nodes in their respective graph domains."""
    kinds, destinations = layout
    if (
        "both" in (*kinds.values(), *destinations)
        and counts["source"] != counts["target"]
    ):
        return False
    capacities = {kind: padded_rows(n, 8) + 1 for kind, n in counts.items()}
    capacities["both"] = capacities["source"]
    capacities["edge"] = padded_rows(values[-1].numel(), ROW_BUCKET)

    def shape(value, kind):
        return (
            (capacities[kind], *value.shape[1:]) if kind in capacities else value.shape
        )

    inputs = values[:-2]
    if any(
        kind != "shared"
        and (
            not inputs[i].ndim
            or inputs[i].shape[0]
            != (
                values[-1].numel()
                if kind == "edge"
                else counts["source" if kind == "both" else kind]
            )
        )
        for i, kind in kinds.items()
    ):
        return False
    edge_shape = (capacities["edge"],)
    execute(
        "edge_program",
        metadata,
        function,
        values,
        outputs,
        shapes=(
            tuple(shape(value, kinds.get(i)) for i, value in enumerate(inputs))
            + (edge_shape, edge_shape),
            tuple(shape(value, kind) for value, kind in zip(outputs, destinations)),
        ),
        fills={
            len(inputs): capacities["source"] - 1,
            len(inputs) + 1: capacities["target"] - 1,
        },
    )
    return True


def edge_program(metadata, function, inputs, source, target, outputs, layout, counts):
    """Replay repeated convolution shapes or independent row-product buckets.

    The caller proves that zero rows contribute nothing to shared outputs.
    Capture only the native evaluation; registered autograd programs still
    build fresh derivative graphs around every invocation.
    """
    if layout is None:
        values = (*inputs, source, target)
        if (
            source.numel() >= MIN_EDGES
            and enabled("edge_program")
            and is_convolution(metadata)
        ):
            from .edge import padding_layout

            indexed = padding_layout(metadata, indexed=True)
            if indexed is None or not indexed_replay(
                metadata, function, values, outputs, indexed, counts
            ):
                execute("edge_program", metadata, function, values, outputs)
        else:
            function(values, outputs)
        return
    if source.numel() < MIN_ROWS or not enabled("product"):
        function((*inputs, source, target), outputs)
        return
    input_kinds, output_kinds = layout
    if any(
        kind == "edge" and inputs[i].shape[0] != source.numel()
        for i, kind in input_kinds.items()
    ):
        function((*inputs, source, target), outputs)
        return
    rows = source.numel()
    # Limit captures as batch sizes grow without padding small batches heavily.
    capacity = padded_rows(rows, ROW_BUCKET)
    execute(
        "product",
        metadata,
        function,
        (*inputs, source, target),
        outputs,
        shapes=(
            tuple(
                (capacity, *value.shape[1:])
                if input_kinds.get(i) == "edge"
                else value.shape
                for i, value in enumerate(inputs)
            )
            + ((capacity,), (capacity,)),
            tuple(
                (capacity, *value.shape[1:]) if kind == "edge" else value.shape
                for value, kind in zip(outputs, output_kinds)
            ),
        ),
    )
