"""Tiled GEMMs and fused pointwise expressions for local interactions."""

from functools import lru_cache

import torch

from ....conv.codegen import source as expression_source
from ....conv.program import Program
from ....kernels.cuda import kernels, runtime
from ....utils import parse_metadata


@lru_cache(maxsize=128)
def execution_plan(metadata):
    nodes, outputs = parse_metadata(metadata)
    result = []
    for op, _, args, data in nodes:
        result.append(
            data[1] != "shared" if op == "input" else any(result[i] for i in args)
        )
    return nodes, outputs, tuple(result), {}


@lru_cache(maxsize=64)
def forward_plan(base):
    nodes, score, value, *_ = base
    return execution_plan(repr((nodes, ((score, 0, "target"), (value, 1, "target")))))


def expression(nodes, root, dependent):
    """Separate dense contractions from the surrounding fused operations."""
    program, leaves, translated = Program(), [], {}

    def visit(i):
        if i not in translated:
            op, size, args, data = nodes[i]
            if op in ("input", "matmul") or not dependent[i]:
                slot = len(leaves)
                leaves.append(i)
                translated[i] = program.input(
                    slot, "edge" if dependent[i] else "shared", size
                )
            else:
                translated[i] = program.add(
                    op, size, tuple(visit(j) for j in args), data
                )
        return translated[i]

    output = visit(root)
    return repr((tuple(program.nodes), ((output, 0, "edge"),))), tuple(leaves)


@lru_cache(maxsize=1024)
def expression_kernel(metadata, dtype, device):
    code, storage, slots = expression_source(dtype, metadata, "edge", 0, 4, 0.0, 0, 1)
    kernel = kernels([code], device)[code]
    size = storage * (8 if dtype == torch.float64 else 4)
    processors = torch.cuda.get_device_properties(device).multi_processor_count
    return kernel, storage, slots, size, processors


@lru_cache(maxsize=256)
def indices(values, device):
    return torch.tensor(values, dtype=torch.int64, device=device)


def transform(op, size, values, data):
    """Evaluate shared expressions and linear maps of reduced adjoints."""
    x = values[0] if values else None
    if op == "add":
        return x + values[1]
    if op == "mul":
        return x * values[1]
    if op == "slice":
        return x[:, data : data + size]
    if op == "unslice":
        return torch.nn.functional.pad(x, (data, size - data - x.shape[1]))
    if op == "concat":
        return torch.cat(values, dim=1)
    if op == "transpose":
        return x.reshape(-1, *data).transpose(1, 2).reshape(x.shape[0], size)
    if op == "gather":
        return x.index_select(1, indices(data, x.device))
    if op == "scatter":
        return x.new_zeros((x.shape[0], size)).index_add_(1, indices(data, x.device), x)
    if op == "matmul":
        rows, inner, columns = data
        a, b = x.reshape(-1, rows, inner), values[1].reshape(-1, inner, columns)
        if b.shape[0] == 1:
            return (a.reshape(-1, inner) @ b[0]).reshape(a.shape[0], size)
        return torch.matmul(a, b).flatten(1)
    if op == "silu":
        return torch.nn.functional.silu(x)
    if op in ("sigmoid", "tanh", "sin", "cos", "exp", "reciprocal"):
        return getattr(torch, op)(x)
    raise NotImplementedError(op)


class TiledProgram:
    """Evaluate one edge tile, sharing parameter expressions across tiles."""

    def __init__(self, plan, inputs, source, target, begin, end, shared):
        self.nodes, _, self.dependent, self.expressions = plan
        self.inputs = inputs
        self.source, self.target = source[begin:end], target[begin:end]
        self.begin, self.end, self.shared = begin, end, shared
        self.values, self.reductions = {}, {}

    def value(self, i):
        cache = self.values if self.dependent[i] else self.shared
        if i not in cache:
            op, size, args, data = self.nodes[i]
            if op == "input":
                slot, kind = data
                x = self.inputs[slot].reshape(-1, size)
                if kind == "edge":
                    x = x[self.begin : self.end]
                elif kind in ("source", "target"):
                    x = x.index_select(
                        0, self.source if kind == "source" else self.target
                    )
            elif op == "constant":
                x = self.inputs[0].new_full((1, size), data)
            elif op == "matmul" or not self.dependent[i]:
                x = transform(op, size, [self.value(j) for j in args], data)
            else:
                if i not in self.expressions:
                    self.expressions[i] = expression(self.nodes, i, self.dependent)
                metadata, leaves = self.expressions[i]
                values = [self.value(j).contiguous() for j in leaves]
                x = self.inputs[0].new_empty((self.end - self.begin, size))
                kernel, storage, slots, nbytes, processors = expression_kernel(
                    metadata, x.dtype, x.device
                )
                shared = nbytes <= 49152
                blocks = x.shape[0] if shared else min(x.shape[0], 2 * processors)
                workspace = None if shared else x.new_empty((blocks, storage))
                args = [v.data_ptr() for v in values[:slots]]
                args += [
                    self.source.data_ptr(),
                    self.target.data_ptr(),
                    0,
                    0,
                    0 if workspace is None else workspace.data_ptr(),
                    x.data_ptr(),
                    x.shape[0],
                ]
                runtime().launch(
                    [
                        (
                            kernel,
                            args,
                            blocks,
                            1,
                            256 if storage >= 8192 else 128,
                            nbytes if shared else 0,
                        )
                    ],
                    torch.cuda.current_stream(x.device).cuda_stream,
                )
            cache[i] = x
        return cache[i]

    def reduce(self, i):
        """Sum parameter adjoints before allocating per-edge outer products."""
        if i not in self.reductions:
            op, size, args, data = self.nodes[i]
            if not self.dependent[i]:
                x = self.value(i) * (self.end - self.begin)
            elif op == "matmul" and all(self.dependent[j] for j in args):
                rows, inner, columns = data
                a = self.value(args[0]).reshape(-1, rows, inner)
                b = self.value(args[1]).reshape(-1, inner, columns)
                x = (
                    a.permute(1, 0, 2).reshape(rows, -1) @ b.reshape(-1, columns)
                ).reshape(1, size)
            elif op in (
                "add",
                "slice",
                "unslice",
                "concat",
                "transpose",
                "gather",
                "scatter",
            ):
                x = transform(op, size, [self.reduce(j) for j in args], data)
            elif op in ("mul", "matmul") and any(not self.dependent[j] for j in args):
                x = transform(
                    op,
                    size,
                    [
                        self.reduce(j) if self.dependent[j] else self.value(j)
                        for j in args
                    ],
                    data,
                )
            else:
                x = self.value(i).sum(0, keepdim=True)
            self.reductions[i] = x
        return self.reductions[i]

    def accumulate(self, i, output):
        """Write parameter slices without padding each path to the full matrix."""
        op, _, args, data = self.nodes[i]
        if op == "add":
            for j in args:
                self.accumulate(j, output)
        elif op == "unslice":
            self.accumulate(args[0], output[:, data : data + self.nodes[args[0]][1]])
        else:
            output.add_(self.reduce(i))


def launch(metadata, inputs, source, target, outputs, tile_size=4096):
    """Evaluate tiled adjoints without full-edge features or weight gradients."""
    plan = execution_plan(metadata)
    _, descriptions, _, _ = plan
    slots = sorted({slot for _, slot, _ in descriptions})
    shared_slots = {slot for _, slot, kind in descriptions if kind == "shared"}
    destinations = {
        slot: output.reshape(1, -1) if slot in shared_slots else output.flatten(1)
        for slot, output in zip(slots, outputs)
    }
    shared = {}
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(plan, inputs, source, target, begin, end, shared)
        for root, slot, kind in descriptions:
            output = destinations[slot]
            if kind == "shared":
                tile.accumulate(root, output)
            elif kind == "edge":
                output[begin:end].add_(tile.value(root))
            else:
                output.index_add_(
                    0,
                    tile.source if kind == "source" else tile.target,
                    tile.value(root),
                )


def forward(base, inputs, source, target, outputs, tile_size=4096):
    """Retain only edge scores between the score and message passes."""
    nodes, score, value, _, heads, channels, eps = base
    plan = forward_plan(base)
    result, denominator, maximum = outputs
    shared = {}
    scores = inputs[0].new_empty((source.numel(), heads))
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(plan, inputs, source, target, begin, end, shared)
        scores[begin:end] = tile.value(score)
    maximum.fill_(-torch.inf)
    maximum.scatter_reduce_(0, target[:, None].expand_as(scores), scores, reduce="amax")
    maximum.masked_fill_(maximum.isneginf(), 0)
    exponential = (scores - maximum[target]).exp()
    denominator.zero_().index_add_(0, target, exponential * inputs[4]).add_(eps)
    scale = exponential * inputs[4].square() / denominator[target]
    result.zero_()
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(plan, inputs, source, target, begin, end, shared)
        message = tile.value(value).reshape(end - begin, -1, heads, channels // heads)
        message = (message * scale[begin:end, None, :, None]).flatten(1)
        result.flatten(1).index_add_(0, tile.target, message)
