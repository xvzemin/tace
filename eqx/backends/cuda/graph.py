"""Bounded CUDA Graph replay of native contractions and their adjoints."""

import os
from collections import OrderedDict
from threading import RLock

import torch

ENABLED = {"o3": True, "o2_o3": True, "ace": True}
EDGE_BUCKET = 2048
MIN_EDGES = 1024
MAX_GRAPHS = 64
MAX_BYTES = 4 << 30
_GRAPHS = OrderedDict()
_LOCK = RLock()


def enabled(name):
    """Return whether native graph replay is enabled for this operator."""
    return os.environ.get("EQX_USE_CUDA_GRAPH", "0") == "1" and ENABLED[name]


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
    if estimated > MAX_BYTES // 2:
        function(inputs, outputs)
        return
    with _LOCK, torch.cuda.device(device):
        cached = _GRAPHS.get(signature)
        if cached is None:
            before = torch.cuda.memory_allocated(device)
            values = [x.new_zeros(shape) for shape, x in zip(input_shapes, inputs)]
            results = [x.new_empty(shape) for shape, x in zip(output_shapes, outputs)]
        else:
            graph, values, results, _ = cached
        for i in read:
            value, x = values[i], inputs[i]
            if value.shape != x.shape:
                value[x.shape[0] :].fill_(fills.get(i, 0))
                value = value[: x.shape[0]]
            value.copy_(x)
        if cached is None:
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(stream)
            with torch.cuda.stream(capture_stream):
                # Compile and initialize every native phase before capture.
                function(values, results)
                function(values, results)
            stream.wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                function(values, results)
            stream.wait_stream(capture_stream)
            size = max(estimated, torch.cuda.memory_allocated(device) - before)
            _GRAPHS[signature] = graph, values, results, size
            while len(_GRAPHS) > 1 and (
                len(_GRAPHS) > MAX_GRAPHS
                or sum(entry[3] for entry in _GRAPHS.values()) > MAX_BYTES
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
