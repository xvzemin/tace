"""Tiled matrix products and fused expressions for convolution programs."""

from collections import Counter
from functools import lru_cache

import torch

from ..kernels.cuda import kernels, runtime
from ..utils.metadata import parse_metadata
from .codegen import source as expression_source
from .program import Program

TILE_BYTES = 64 << 20
TILE_SIZE = 131072


def node_projection(nodes, i):
    op, _, args, data = nodes[i]
    return (
        op == "matmul"
        and data[0] == 1
        and nodes[args[0]][0] == "input"
        and nodes[args[0]][3][1] in ("source", "target")
    )


@lru_cache(maxsize=128)
def execution_plan(metadata):
    nodes, outputs = parse_metadata(metadata)
    result = []
    for op, _, args, data in nodes:
        result.append(
            data[1] != "shared" if op == "input" else any(result[i] for i in args)
        )
    expressions, dependencies, uses = {}, {}, Counter()

    def require(kind, i):
        key = kind, i
        uses[key] += 1
        if key in dependencies:
            return
        op, _, args, _ = nodes[i]
        if kind == "value":
            if op in ("input", "constant"):
                children = ()
            elif node_projection(nodes, i):
                children = (("value", args[1]),)
            elif op == "convolution":
                output = nodes[i][3][2]
                children = tuple(
                    ("value", j)
                    for role, j in enumerate(args)
                    if role != output
                    and not (
                        nodes[j][0] == "input"
                        and nodes[j][3][1] in ("source", "target")
                    )
                )
            elif op == "matmul" or not result[i]:
                children = tuple(("value", j) for j in args)
            else:
                expressions[i] = expression(nodes, i, result)
                children = tuple(("value", j) for j in expressions[i][1])
        elif not result[i]:
            children = (("value", i),)
        elif op == "convolution":
            children = (("value", i),)
        elif op == "matmul" and all(result[j] for j in args):
            children = tuple(("value", j) for j in args)
        elif op in (
            "add",
            "slice",
            "unslice",
            "concat",
            "transpose",
            "gather",
            "scatter",
        ):
            children = tuple(("reduce", j) for j in args)
        elif op in ("mul", "matmul") and any(not result[j] for j in args):
            children = tuple(("reduce" if result[j] else "value", j) for j in args)
        else:
            children = (("value", i),)
        dependencies[key] = children
        for child in children:
            require(*child)

    def accumulate(i):
        op, _, args, _ = nodes[i]
        if op in ("add", "unslice"):
            for j in args:
                accumulate(j)
        else:
            require("reduce", i)

    for root, _, kind in outputs:
        if kind == "shared":
            accumulate(root)
        else:
            require("value", root)
    return nodes, outputs, tuple(result), expressions, dependencies, uses


@lru_cache(maxsize=128)
def uses_matrix_products(metadata):
    """Select tiled execution for dense maps and convolution expressions."""
    nodes, _ = parse_metadata(metadata)
    return any(
        op in ("convolution", "normalize", "inv_norm", "mean", "rsqrt")
        for op, _, _, _ in nodes
    ) or any(
        op == "matmul"
        and data[0] == 1
        and nodes[args[1]][0] == "input"
        and nodes[args[1]][3][1] == "shared"
        for op, _, args, data in nodes
    )


def expression(nodes, root, dependent):
    """Separate dense contractions from the surrounding fused operations."""
    program, leaves, translated = Program(), [], {}

    def visit(i):
        if i not in translated:
            op, size, args, data = nodes[i]
            if (
                op
                in (
                    "input",
                    "matmul",
                    "convolution",
                )
                or not dependent[i]
            ):
                slot = len(leaves)
                leaves.append(i)
                kind = (
                    data[1]
                    if op == "input"
                    else "shared"
                    if not dependent[i]
                    else nodes[args[0]][3][1]
                    if node_projection(nodes, i)
                    else "edge"
                )
                translated[i] = program.input(slot, kind, size)
            else:
                translated[i] = program.add(
                    op, size, tuple(visit(j) for j in args), data
                )
        return translated[i]

    output = visit(root)
    return repr((tuple(program.nodes), ((output, 0, "edge"),))), tuple(leaves)


@lru_cache(maxsize=1024)
def expression_kernel(metadata, dtype, device):
    nodes, _ = parse_metadata(metadata)
    mode = (
        "row"
        if any(op in ("normalize", "inv_norm", "mean") for op, _, _, _ in nodes)
        else "edge"
        if any(op in ("product", "scatter", "matmul") for op, _, _, _ in nodes)
        else "pointwise"
    )
    code, storage, slots = expression_source(dtype, metadata, mode, 0, 4, 0.0, 0, 1)
    kernel = kernels([code], device)[code]
    size = storage * (8 if dtype == torch.float64 else 4)
    processors = torch.cuda.get_device_properties(device).multi_processor_count
    return kernel, storage, slots, size, processors, mode


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
        return x.reshape(x.shape[0], *data).transpose(1, 2).reshape(x.shape[0], size)
    if op == "gather":
        return x.index_select(1, indices(data, x.device))
    if op == "scatter":
        return x.new_zeros((x.shape[0], size)).index_add_(1, indices(data, x.device), x)
    if op == "matmul":
        rows, inner, columns = data
        a = x.reshape(x.shape[0], rows, inner)
        b = values[1].reshape(values[1].shape[0], inner, columns)
        if b.shape[0] == 1:
            return (a.reshape(a.shape[0] * rows, inner) @ b[0]).reshape(
                a.shape[0], size
            )
        return torch.matmul(a, b).flatten(1)
    if op == "silu":
        return torch.nn.functional.silu(x)
    if op == "mean":
        return x.mean(-1, keepdim=True).expand_as(x)
    if op in ("sigmoid", "tanh", "sin", "cos", "exp", "reciprocal", "rsqrt"):
        return getattr(torch, op)(x)
    raise NotImplementedError(op)


class TiledProgram:
    """Evaluate one edge tile, sharing parameter expressions across tiles."""

    def __init__(self, plan, inputs, source, target, begin, end, shared):
        self.nodes, _, self.dependent, self.expressions, self.dependencies, uses = plan
        self.uses = uses.copy()
        self.inputs = inputs
        self.source, self.target = source[begin:end], target[begin:end]
        self.begin, self.end, self.shared = begin, end, shared
        self.values, self.reductions = {}, {}
        self.destinations = {}
        self.completed = set()
        self.local = None

    def release(self, kind, i):
        """Release a tile intermediate after its last consumer is enqueued."""
        key = kind, i
        self.uses[key] -= 1
        if not self.uses[key]:
            (self.values if kind == "value" else self.reductions).pop(i, None)

    def project_node(self, i):
        """Cache a shared linear map before gathering its node rows."""
        _, _, args, data = self.nodes[i]
        if ("node", i) not in self.shared:
            slot, _ = self.nodes[args[0]][3]
            weight = self.value(args[1]).reshape(data[1], data[2])
            self.shared["node", i] = self.inputs[slot].flatten(1) @ weight
        return self.shared["node", i]

    def value(self, i):
        cache = self.values if self.dependent[i] else self.shared
        if i not in cache:
            op, size, args, data = self.nodes[i]
            if op == "input":
                slot, kind = data
                x = (
                    self.inputs[slot].reshape(1, -1)
                    if kind == "shared"
                    else self.inputs[slot].flatten(1)
                )
                if kind == "edge":
                    x = x[self.begin : self.end]
                elif kind in ("source", "target"):
                    x = x.index_select(
                        0, self.source if kind == "source" else self.target
                    )
            elif op == "constant":
                x = self.inputs[0].new_full((1, size), data)
            elif node_projection(self.nodes, i):
                _, kind = self.nodes[args[0]][3]
                x = self.project_node(i).index_select(
                    0, self.source if kind == "source" else self.target
                )
            elif op == "convolution":
                x = self.convolution(i)
            elif op in ("normalize", "inv_norm") and not self.dependent[i]:
                key = ("normalization", args[0], data)
                if key not in self.values:
                    value = self.value(args[0])
                    if data[1]:
                        normal, _, inverse = torch.native_layer_norm(
                            value, (size,), None, None, data[0]
                        )
                    else:
                        inverse = (
                            value.square().mean(-1, keepdim=True) + data[0]
                        ).rsqrt()
                        normal = value * inverse
                    self.values[key] = normal, inverse.expand_as(value)
                x = self.values[key][op == "inv_norm"]
            elif op == "matmul" or not self.dependent[i]:
                x = transform(op, size, [self.value(j) for j in args], data)
            else:
                if i not in self.expressions:
                    self.expressions[i] = expression(self.nodes, i, self.dependent)
                metadata, leaves = self.expressions[i]
                values = [
                    (
                        self.inputs[self.nodes[j][3][0]]
                        if self.nodes[j][0] == "input"
                        and self.nodes[j][3][1] in ("source", "target")
                        else self.project_node(j)
                        if node_projection(self.nodes, j)
                        else self.value(j)
                    ).contiguous()
                    for j in leaves
                ]
                x = self.inputs[0].new_empty((self.end - self.begin, size))
                kernel, storage, slots, nbytes, processors, mode = expression_kernel(
                    metadata, x.dtype, x.device
                )
                shared = nbytes <= 49152
                blocks = x.shape[0] if shared else min(x.shape[0], 2 * processors)
                if mode == "pointwise":
                    blocks = min((x.numel() + 255) // 256, 16 * processors)
                elif mode == "row":
                    blocks = (x.shape[0] + 3) // 4
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
                            256 if mode == "pointwise" or storage >= 8192 else 128,
                            nbytes if shared else 0,
                        )
                    ],
                    torch.cuda.current_stream(x.device).cuda_stream,
                )
            cache[i] = x
            if op != "convolution":
                for child in self.dependencies["value", i]:
                    self.release(*child)
        return cache[i]

    def convolution(self, i, destination=None):
        """Dispatch a tensor-product tile without gathering node features."""
        if i in self.values:
            return self.values[i]
        _, _, args, data = self.nodes[i]
        kind, metadata, _, weighted, _, order = data
        siblings = [
            j
            for j, (op, _, other, spec) in enumerate(self.nodes)
            if op == "convolution"
            and (j == i or order <= 1)
            and ("value", j) in self.dependencies
            and j not in self.completed
            and spec[:2] == data[:2]
            and spec[3:] == data[3:]
            and other == args
            and (
                spec[2] not in (0, 4 if kind == "o3" else 6)
                or j in self.destinations
                or j == i
            )
        ]
        groups, results = {}, []
        for j in siblings:
            source, target, values, result = self.convolution_operands(
                j, self.destinations.get(j, destination if j == i else None)
            )
            role = self.nodes[j][3][2]
            key = source.data_ptr(), target.data_ptr()
            if key not in groups:
                groups[key] = (source, target, [], values, [])
            _, _, roles, operands, outputs = groups[key]
            # Each VJP ignores its own operand. Other VJPs supply that operand,
            # allowing all requested derivatives to share one contraction.
            for k, value in enumerate(values):
                if k != role:
                    operands[k] = value
            roles.append(role)
            outputs.append(result)
            results.append(result.reshape(1, -1) if role == 2 else result)
        for source, target, roles, operands, outputs in groups.values():
            if kind == "o3":
                from .o3.cuda import contract

                program = (
                    (
                        tuple(range(len(operands))),
                        weighted,
                        tuple(zip(roles, range(len(roles)))),
                    ),
                )
                contract(metadata, program, source, target, operands, outputs)
            else:
                from .o2_o3.convolution import kernel_plan
                from .o2_o3.cuda import contract_many

                contract_many(
                    kernel_plan(metadata),
                    source,
                    target,
                    [(tuple(roles), tuple(operands), tuple(outputs), weighted)],
                )
        for j, result in zip(siblings, results):
            self.values[j] = result
            self.completed.add(j)
            for child in self.dependencies["value", j]:
                self.release(*child)
        return self.values[i]

    def convolution_operands(self, i, destination):
        """Prepare a tile's operands and the requested reduction destination."""
        _, size, args, (kind, metadata, role, weighted, shape, _) = self.nodes[i]
        output_role = 4 if kind == "o3" else 6
        source, target = self.source, self.target
        rows = self.end - self.begin
        values = []
        for k, j in enumerate(args):
            if k == role:
                if k == 2:
                    value = self.inputs[0].new_empty(shape)
                elif destination is not None:
                    value = destination
                else:
                    value = self.inputs[0].new_empty((rows, size))
                    if k in (0, output_role):
                        if self.local is None:
                            self.local = torch.arange(rows, device=source.device)
                        if k == 0:
                            source = self.local
                        else:
                            target = self.local
                values.append(value)
                continue
            op, width, _, data = self.nodes[j]
            if op == "input" and data[1] in ("source", "target"):
                value = self.inputs[data[0]].flatten(1)
            else:
                value = self.value(j)
                if k not in (0, 2, output_role) and not self.dependent[j]:
                    # Keep adjoints edge-wise until the expression program
                    # reduces them, including broadcast radial inputs.
                    value = value.expand(rows, -1).contiguous()
                if k == 0:
                    value = value.expand(rows, -1).contiguous()
                    if self.local is None:
                        self.local = torch.arange(rows, device=source.device)
                    source = self.local
                elif k == output_role:
                    value = value.expand(rows, -1).contiguous()
                    if self.local is None:
                        self.local = torch.arange(rows, device=source.device)
                    target = self.local
            if k == 2:
                cache = self.values if self.dependent[j] else self.shared
                key = ("projection", j, shape)
                if key not in cache:
                    cache[key] = value.reshape(shape)
                value = cache[key]
            values.append(value.contiguous())
        result = values[role]
        if destination is None:
            result = torch.zeros_like(result)
        return source, target, values, result

    def reduce(self, i):
        """Sum parameter adjoints before allocating per-edge outer products."""
        if i not in self.reductions:
            op, size, args, data = self.nodes[i]
            if not self.dependent[i]:
                x = self.value(i) * (self.end - self.begin)
            elif op == "convolution":
                # The contraction already reduces shared projection adjoints.
                x = self.value(i)
                if data[2] != 2:
                    x = x.sum(0, keepdim=True)
            elif op == "matmul" and all(self.dependent[j] for j in args):
                rows, inner, columns = data
                a, b = self.value(args[0]), self.value(args[1])
                a = a.reshape(a.shape[0], rows, inner)
                b = b.reshape(b.shape[0], inner, columns)
                x = (
                    a.permute(1, 0, 2).reshape(rows, a.shape[0] * inner)
                    @ b.reshape(b.shape[0] * inner, columns)
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
            for child in self.dependencies["reduce", i]:
                self.release(*child)
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
            self.release("reduce", i)


def launch(metadata, inputs, source, target, outputs, tile_size=None):
    """Evaluate a program with bounded intermediate edge storage."""
    plan = execution_plan(metadata)
    if tile_size is None:
        width = max(
            (max(data[1:]) for op, _, _, data in plan[0] if op == "matmul"), default=128
        )
        tile_size = max(
            256,
            min(TILE_SIZE, TILE_BYTES // (max(1, width) * inputs[0].element_size())),
        )
    descriptions = plan[1]
    writes = Counter(slot for _, slot, _ in descriptions)
    slots = sorted({slot for _, slot, _ in descriptions})
    shared_slots = {slot for _, slot, kind in descriptions if kind == "shared"}
    destinations = {
        slot: output.reshape(1, -1) if slot in shared_slots else output.flatten(1)
        for slot, output in zip(slots, outputs)
    }
    shared, node_adjoints = {}, {}
    for begin in range(0, source.numel(), tile_size):
        end = min(begin + tile_size, source.numel())
        tile = TiledProgram(plan, inputs, source, target, begin, end, shared)
        tile.destinations = {
            root: destinations[slot]
            for root, slot, kind in descriptions
            if kind in ("source", "target") and tile.nodes[root][0] == "convolution"
        }
        for root, slot, kind in descriptions:
            output = destinations[slot]
            if kind == "shared":
                tile.accumulate(root, output)
            elif kind == "edge":
                if writes[slot] == 1:
                    output[begin:end].copy_(tile.value(root))
                else:
                    output[begin:end].add_(tile.value(root))
                tile.release("value", root)
            else:
                op, _, args, data = tile.nodes[root]
                if (
                    op == "matmul"
                    and data[0] == 1
                    and not tile.dependent[args[1]]
                    and plan[-1]["value", root] == 1
                ):
                    # Sum the narrow adjoint at nodes before the shared linear
                    # map, instead of expanding it on every incident edge.
                    key = root, slot, kind
                    if key not in node_adjoints:
                        node_adjoints[key] = (
                            output.new_zeros((output.shape[0], data[1])),
                            tile.value(args[1]).reshape(data[1:]),
                        )
                    node_adjoints[key][0].index_add_(
                        0,
                        tile.source if kind == "source" else tile.target,
                        tile.value(args[0]),
                    )
                    for child in tile.dependencies["value", root]:
                        tile.release(*child)
                elif op == "convolution":
                    tile.convolution(root, output)
                else:
                    output.index_add_(
                        0,
                        tile.source if kind == "source" else tile.target,
                        tile.value(root),
                    )
                tile.release("value", root)
    for (_, slot, _), (adjoint, weight) in node_adjoints.items():
        destinations[slot].addmm_(adjoint, weight)
