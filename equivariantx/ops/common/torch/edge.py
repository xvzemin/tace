"""PyTorch evaluation of edge expressions."""

from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.metadata import column_schedule, parse_metadata
from ....utils.outputs import edge_node_counts
from ....utils.program import scalar_product
from ... import config
from ..edge import edge_program
from ..pointwise import activation_vjp
from .product import product


def evaluate(metadata, inputs, source, target, num_nodes):
    """Evaluate the optimized Torch schedule under the internal memory policy."""
    if config.TORCH_RECOMPUTE:
        return evaluate_recomputed(metadata, inputs, source, target, num_nodes)
    return evaluate_torch(metadata, inputs, source, target, num_nodes, optimized=True)


def evaluate_recomputed(metadata, inputs, source, target, num_nodes):
    """Evaluate a differentiable Torch program while saving only its operands."""
    return list(edge_program("torch", metadata, num_nodes, source, target, *inputs))


@lru_cache(maxsize=64)
def program_indices(metadata, device, selected=None):
    """Cache only the column maps that need indexed kernels."""
    nodes, _ = parse_metadata(metadata)
    if selected is None:
        selected = [
            i for i, (op, *_) in enumerate(nodes) if op in ("gather", "scatter")
        ]
    return {
        i: torch.tensor(nodes[i][3], device=device, dtype=torch.long) for i in selected
    }


def channel_gather(value, schedule):
    """Broadcast fixed channel sections without indexed-gradient atomics."""
    start, width, repeats, interleaved = schedule
    value = value[:, start : start + width]
    if repeats == 1:
        return value
    return (
        value[:, :, None].expand(-1, -1, repeats)
        if interleaved
        else value[:, None, :].expand(-1, repeats, -1)
    ).reshape(value.shape[0], width * repeats)


def channel_scatter(value, schedule, size):
    """Sum repeated channel sections before restoring their output offset."""
    start, width, repeats, interleaved = schedule
    if repeats != 1:
        value = (
            value.reshape(value.shape[0], width, repeats).sum(2)
            if interleaved
            else value.reshape(value.shape[0], repeats, width).sum(1)
        )
    return torch.nn.functional.pad(value, (start, size - start - width))


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def program_schedule(metadata, static_columns):
    """Push shared reductions through linear operations before evaluating them."""
    nodes, outputs = parse_metadata(metadata)
    shared = []
    whole_rows = {root for root, _, _ in outputs}
    for op, _, args, data in nodes:
        shared.append(
            data[1] == "shared" if op == "input" else all(shared[arg] for arg in args)
        )
        if op != "slice":
            whole_rows.update(args)
    order, visited, users = [], set(), {}

    def visit(node, reduce=False):
        key = node, reduce
        if key in visited:
            return key
        op, _, args, _ = nodes[node]
        action = op
        broadcast = None
        if static_columns and not reduce and op == "mul":
            for side, arg in enumerate(args):
                name, _, parent, indices = nodes[arg]
                if name == "gather":
                    schedule = column_schedule(indices, nodes[parent[0]][1])
                    if schedule is not None and schedule[2] > 1:
                        broadcast = (side, parent[0], schedule)
                        break
        factor = scalar_product(nodes, args) if op == "mul" else None
        if factor is not None and (not reduce or not shared[node]):
            arg, scale = factor
            action, dependencies = ("scale", scale), (visit(arg, reduce),)
        elif (
            not reduce
            and op == "slice"
            and args[0] not in whole_rows
            and nodes[args[0]][0] == "input"
            and nodes[args[0]][3][1] in ("source", "target")
        ):
            action = ("node_slice", *nodes[args[0]][3])
            dependencies = ()
        elif broadcast is not None:
            side, parent, schedule = broadcast
            action = ("channel_mul", *schedule)
            dependencies = (visit(args[1 - side]), visit(parent))
        elif op == "matmul" and (not reduce or not shared[node]):
            rows, inner, columns = nodes[node][3]
            transposed = tuple(
                nodes[arg][0] == "transpose" and nodes[arg][3] == shape
                for arg, shape in zip(args, ((inner, rows), (columns, inner)))
            )
            action = ("matmul", *transposed)
            dependencies = tuple(
                visit(nodes[arg][2][0] if transpose else arg)
                for arg, transpose in zip(args, transposed)
            )
        elif not reduce or op == "matmul" and not shared[node]:
            dependencies = tuple(visit(arg) for arg in args)
        elif shared[node]:
            action, dependencies = "sum_shared", (visit(node),)
        elif op in (
            "add",
            "slice",
            "unslice",
            "concat",
            "transpose",
            "gather",
            "scatter",
        ):
            dependencies = tuple(visit(arg, True) for arg in args)
        elif op == "mul" and any(shared[arg] for arg in args):
            dependencies = tuple(visit(arg, not shared[arg]) for arg in args)
        else:
            action, dependencies = "sum_edges", (visit(node),)
        if static_columns and action in ("gather", "scatter"):
            width = nodes[args[0]][1] if action == "gather" else nodes[node][1]
            schedule = column_schedule(nodes[node][3], width)
            if schedule is not None:
                action = ("channel_" + action, schedule)
        for dependency in dependencies:
            users[dependency] = users.get(dependency, 0) + 1
        visited.add(key)
        order.append((key, action, dependencies))
        return key

    roots = tuple(visit(root, kind == "shared") for root, _, kind in outputs)
    destinations = {}
    for index, key in enumerate(roots):
        destinations.setdefault(key, []).append(index)
        users[key] = users.get(key, 0) + 1
    return (
        tuple(order),
        tuple(shared),
        tuple(users.items()),
        roots,
        tuple((key, tuple(indices)) for key, indices in destinations.items()),
        tuple(
            sorted(
                {key[0] for key, action, _ in order if action in ("gather", "scatter")}
            )
        ),
    )


def matrix_product(left, right, shape, shared, edges, reduce, transposed):
    """Use GEMM for shared matrices and sum outer products before materializing."""
    rows, inner, columns = shape
    left_shared, right_shared = shared
    left_rows, right_rows = (
        (1 if left_shared else edges),
        (1 if right_shared else edges),
    )
    left = (
        left.reshape(left_rows, inner, rows).transpose(1, 2)
        if transposed[0]
        else left.reshape(left_rows, rows, inner)
    )
    right = (
        right.reshape(right_rows, columns, inner).transpose(1, 2)
        if transposed[1]
        else right.reshape(right_rows, inner, columns)
    )
    if reduce:
        if left_shared and right_shared:
            result = (left[0] @ right[0]) * edges
        elif left_shared:
            result = left[0] @ right.sum(0)
        elif right_shared:
            result = left.sum(0) @ right[0]
        else:
            result = left.permute(1, 0, 2).reshape(rows, edges * inner) @ right.reshape(
                edges * inner, columns
            )
        return result.reshape(1, rows * columns)
    if inner == 1 and (rows == 1 or columns == 1):
        return (left * right).flatten(1)
    if right_shared:
        return (left.reshape(left_rows * rows, inner) @ right[0]).reshape(
            left_rows, rows * columns
        )
    if left_shared:
        return torch.matmul(left[0], right).reshape(right_rows, rows * columns)
    return torch.bmm(left, right).reshape(edges, rows * columns)


def convolution(data, operands, rows):
    """Evaluate embedded edge-local contractions, including projection adjoints."""
    kind, metadata, role, weighted, projection_shape, _ = data
    if rows == 0:
        return operands[0].new_zeros((0, operands[role].shape[1])) + sum(
            value[..., :0].sum() for value in operands
        )
    if kind == "o3":
        from .o3 import contraction
    elif kind == "o2":
        from .convolution import contraction
    else:
        raise NotImplementedError(f"Torch convolution kind {kind!r} is unavailable.")
    program = repr(((tuple(range(len(operands))), weighted, ((role, 0),)),))
    index = torch.arange(rows, dtype=torch.long, device=operands[0].device)
    values = [value.expand(rows, -1) for value in operands]
    # Projection transposes keep one matrix per edge until the enclosing
    # expression reduces shared parameters. No nested AD transform is needed.
    values[2] = values[2].reshape(rows, *projection_shape)
    return contraction(metadata, program, index, index, values)[0].reshape(
        rows, operands[role].shape[1]
    )


def collect_output(value, kind, shared, optimized, source, target, counts):
    """Reduce a completed edge block before evaluating the next output."""
    if kind == "shared":
        return value if optimized else value.sum(0, keepdim=True)
    if shared:
        value = value.expand(source.numel(), -1)
    if kind == "edge":
        return value
    index = source if kind == "source" else target
    return value.new_zeros((counts[kind], value.shape[1])).index_add(0, index, value)


@lru_cache(maxsize=256)
def tile_layout(metadata, static_columns):
    """Return edge input/output slots and the widest live row operand."""
    nodes, outputs = parse_metadata(metadata)
    order, shared, _, _, _, _ = program_schedule(metadata, static_columns)
    inputs, kinds = {}, {}
    for op, _, _, data in nodes:
        if op == "input":
            inputs.setdefault(data[0], set()).add(data[1])
    width = 1
    for (i, reduced), _, _ in order:
        if not reduced and not shared[i]:
            width = max(width, nodes[i][1])
    for _, slot, kind in outputs:
        kinds.setdefault(slot, set()).add(kind)
    if any(
        "edge" in values and len(values) != 1
        for values in (*inputs.values(), *kinds.values())
    ):
        return (), (), 0
    return (
        frozenset(slot for slot, values in inputs.items() if "edge" in values),
        tuple("edge" in kinds[slot] for slot in sorted(kinds)),
        width,
    )


def evaluate_tiled(metadata, inputs, source, target, num_nodes):
    """Evaluate operand-only forward or adjoint programs in bounded edge blocks."""
    # During symbolic capture retain a single dynamic ATen graph, rather
    # than specializing its length or unrolling a Python loop over the graph.
    if (
        not inputs[0].is_cuda
        or not config.TORCH_EDGE_TILE_BYTES
        or torch.compiler.is_compiling()
        or is_fake(inputs[0])
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    ):
        return evaluate_torch(
            metadata, inputs, source, target, num_nodes, optimized=True
        )
    edge_inputs, edge_outputs, width = tile_layout(
        metadata, config.TORCH_STATIC_COLUMNS
    )
    edges = source.numel()
    tile = max(
        1, config.TORCH_EDGE_TILE_BYTES // (max(1, width) * inputs[0].element_size())
    )
    if not width or edges <= tile:
        return evaluate_torch(
            metadata, inputs, source, target, num_nodes, optimized=True
        )
    # Balance the tail without exceeding the selected row budget.
    count = (edges + tile - 1) // tile
    tile = (edges + count - 1) // count
    result = []
    for begin in range(0, edges, tile):
        end = min(begin + tile, edges)
        values = tuple(
            x[begin:end] if i in edge_inputs else x for i, x in enumerate(inputs)
        )
        parts = evaluate_torch(
            metadata,
            values,
            source[begin:end],
            target[begin:end],
            num_nodes,
            optimized=True,
        )
        for i, (part, edge) in enumerate(zip(parts, edge_outputs)):
            if begin == 0:
                result.append(part.new_empty((edges, part.shape[1])) if edge else part)
            if edge:
                result[i][begin:end].copy_(part)
            elif begin:
                # A node/shared result can alias an input (e.g. sum of a
                # constant shared vector). Never mutate that input.
                result[i] = result[i] + part
        del parts
    return result


def evaluate_torch(metadata, inputs, source, target, num_nodes, *, optimized=False):
    """Evaluate an edge program with PyTorch operations and ordinary autograd."""
    nodes, outputs = parse_metadata(metadata)
    edges = source.numel()
    counts = edge_node_counts(metadata, inputs, num_nodes)
    tracing = (
        torch.compiler.is_compiling()
        or is_fake(inputs[0])
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    )
    index_function = program_indices.__wrapped__ if tracing else program_indices
    selected = None
    if optimized:
        order, shared, users, roots, destinations, selected = program_schedule(
            metadata, config.TORCH_STATIC_COLUMNS
        )
        users = dict(users)
        destinations = dict(destinations)
    else:
        order = tuple(
            ((i, False), op, tuple((arg, False) for arg in args))
            for i, (op, _, args, _) in enumerate(nodes)
        )
        shared = (False,) * len(nodes)
        roots = tuple((root, False) for root, _, _ in outputs)
    indices = index_function(metadata, inputs[0].device, selected)
    values, parts = {}, {}
    for key, op, dependencies in order:
        node, reduce = key
        _, size, args, data = nodes[node]
        rows = 1 if shared[node] or reduce else edges
        operands = [values[i] for i in dependencies]
        if op == "sum_shared":
            value = operands[0] * edges
        elif op == "sum_edges":
            value = operands[0].sum(0, keepdim=True)
        elif op == "input":
            slot, kind = data
            value = inputs[slot].flatten(1)
            if kind in ("source", "target"):
                value = value[source if kind == "source" else target]
            elif kind == "shared" and not optimized:
                value = value.expand(edges, -1)
        elif isinstance(op, tuple) and op[0] == "node_slice":
            _, slot, kind = op
            index = source if kind == "source" else target
            value = (
                inputs[slot].flatten(1)[:, data : data + size].index_select(0, index)
            )
        elif op == "constant":
            value = inputs[0].new_full((), data).expand(rows, size)
        elif op == "add":
            value = operands[0] + operands[1]
        elif op == "mul":
            value = operands[0] * operands[1]
        elif isinstance(op, tuple) and op[0] == "scale":
            value = operands[0] * op[1]
        elif isinstance(op, tuple) and op[0] == "channel_mul":
            _, start, width, repeats, interleaved = op
            other, weights = operands
            weights = weights[:, start : start + width]
            if interleaved:
                value = (
                    other.reshape(other.shape[0], width, repeats) * weights[:, :, None]
                )
            else:
                value = (
                    other.reshape(other.shape[0], repeats, width) * weights[:, None, :]
                )
            value = value.flatten(1)
        elif op in ("normalize", "inv_norm"):
            x = operands[0]
            if data[1]:
                x = x - x.mean(-1, keepdim=True)
            inverse = (x.square().mean(-1, keepdim=True) + data[0]).rsqrt()
            value = x * inverse if op == "normalize" else inverse.expand_as(x)
        elif op == "mean":
            value = operands[0].mean(-1, keepdim=True).expand_as(operands[0])
        elif op in ("exp", "sin", "cos", "tanh", "sigmoid", "reciprocal", "rsqrt"):
            value = getattr(torch, op)(operands[0])
        elif op == "silu":
            value = torch.nn.functional.silu(operands[0])
        elif op == "activation_vjp":
            value = activation_vjp(*operands, *data)
        elif op == "slice":
            value = operands[0][:, data : data + size]
        elif op == "unslice":
            value = torch.nn.functional.pad(
                operands[0], (data, size - data - operands[0].shape[-1])
            )
        elif op == "concat":
            value = (
                torch.cat([value.expand(rows, -1) for value in operands], dim=-1)
                if operands
                else inputs[0].new_empty((rows, 0))
            )
        elif isinstance(op, tuple) and op[0] == "channel_gather":
            value = channel_gather(operands[0], op[1])
        elif isinstance(op, tuple) and op[0] == "channel_scatter":
            value = channel_scatter(operands[0], op[1], size)
        elif op == "gather":
            value = operands[0].index_select(1, indices[node])
        elif op == "scatter":
            value = (
                inputs[0]
                .new_zeros((rows, size))
                .index_add(1, indices[node], operands[0])
            )
        elif op == "transpose":
            height, columns = data
            value = (
                operands[0].reshape(rows, height, columns).transpose(1, 2).flatten(1)
            )
        elif isinstance(op, tuple) and op[0] == "matmul":
            value = matrix_product(
                *operands, data, tuple(shared[i] for i in args), edges, reduce, op[1:]
            )
        elif op == "matmul":
            height, width, columns = data
            value = torch.bmm(
                operands[0].reshape(edges, height, width),
                operands[1].reshape(edges, width, columns),
            ).flatten(1)
        elif op == "convolution":
            value = convolution(data, operands, rows)
        elif op == "frame":
            layout, role = data
            ni, no, nw, entries = parse_metadata(layout)
            operands = [value.expand(rows, -1) for value in operands]
            value = inputs[0].new_zeros((rows, (ni, nw, no)[role]))
            for start, mul, dim, offset, row_map in entries:
                for local, row, sign in row_map:
                    matrix_slice = slice(offset + row * dim, offset + (row + 1) * dim)
                    node_slice = slice(start, start + dim * mul)
                    edge_slice = slice(local, local + mul)
                    if role == 2:
                        term = (
                            operands[1][:, matrix_slice, None]
                            * operands[0][:, node_slice].reshape(rows, dim, mul)
                        ).sum(1)
                        section = edge_slice
                    elif role == 0:
                        term = (
                            operands[1][:, matrix_slice, None]
                            * operands[2][:, None, edge_slice]
                        ).flatten(1)
                        section = node_slice
                    else:
                        term = (
                            operands[0][:, node_slice].reshape(rows, dim, mul)
                            * operands[2][:, None, edge_slice]
                        ).sum(-1)
                        section = matrix_slice
                    value[:, section] = value[:, section] + sign * term
        elif op == "product" and optimized:
            value = product(*data, operands, rows)
        elif op == "product":
            (width, dims, paths), role, required = data
            parts = [[] for _ in range(dims[role])]
            for components, coefficient in paths:
                if components[role] < 0 or any(
                    index < 0
                    for k, index in enumerate(components)
                    if required & (1 << k)
                ):
                    continue
                term = None
                for k, index in enumerate(components):
                    if k != role and index >= 0:
                        factor = operands[k][:, index * width : (index + 1) * width]
                        term = factor if term is None else term * factor
                term = (
                    inputs[0].new_full((), coefficient).expand(rows, width)
                    if term is None
                    else term * coefficient
                )
                parts[components[role]].append(term)
            value = torch.cat(
                [
                    sum(part[1:], part[0]).expand(rows, width)
                    if part
                    else inputs[0].new_zeros((rows, width))
                    for part in parts
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError(f"PyTorch edge operation {op!r} is unavailable.")
        values[key] = (
            value.expand(rows, size)
            if optimized and value.shape != (rows, size)
            else value
        )
        if optimized:
            for output in destinations.get(key, ()):
                root, _, kind = outputs[output]
                parts[output] = collect_output(
                    values[key], kind, shared[root], True, source, target, counts
                )
                users[key] -= 1
            if not users.get(key, 0):
                del values[key]
            for arg in dependencies:
                users[arg] -= 1
                if users[arg] == 0:
                    del values[arg]
            del value, operands
    result = {}
    used = (
        {nodes[key[0]][3][0] for key, action, _ in order if action == "input"}
        | {
            action[1]
            for _, action, _ in order
            if isinstance(action, tuple) and action[0] == "node_slice"
        }
        if optimized
        else set()
    )
    unused = tuple(value for slot, value in enumerate(inputs) if slot not in used)
    zero = sum(value[..., :0].sum() for value in unused)
    for index, ((root, slot, kind), key) in enumerate(zip(outputs, roots)):
        value = (
            parts[index]
            if optimized
            else collect_output(
                values[key], kind, shared[root], False, source, target, counts
            )
        )
        result[slot] = result[slot] + value if slot in result else value
    # Attach zero derivatives after aggregation, without copying a whole
    # edge-by-channel tensor solely because an optional operand is unused.
    return [result[slot] + zero if unused else result[slot] for slot in sorted(result)]
