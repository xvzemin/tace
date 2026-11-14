"""PyTorch evaluation of edge expressions."""

from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.metadata import parse_metadata
from ....utils.outputs import edge_node_counts
from ... import config
from ..edge import edge_program
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
def program_indices(metadata, device):
    """Cache fixed column indices on the operand device."""
    nodes, _ = parse_metadata(metadata)
    return {
        i: torch.tensor(data, device=device, dtype=torch.long)
        for i, (op, _, _, data) in enumerate(nodes)
        if op in ("gather", "scatter")
    }


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def column_schedule(indices, size):
    """Identify contiguous channel broadcasts and their reduction adjoints."""
    if not indices:
        return None
    start, stop = min(indices), max(indices) + 1
    width = stop - start
    if start < 0 or stop > size or len(indices) % width:
        return None
    repeats = len(indices) // width
    if indices == tuple(range(start, stop)) * repeats:
        return start, width, repeats, False
    if indices == tuple(i for i in range(start, stop) for _ in range(repeats)):
        return start, width, repeats, True
    return None


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
def program_schedule(metadata):
    """Push shared reductions through linear operations before evaluating them."""
    nodes, outputs = parse_metadata(metadata)
    shared = []
    for op, _, args, data in nodes:
        shared.append(
            data[1] == "shared" if op == "input" else all(shared[arg] for arg in args)
        )
    order, visited, users = [], set(), {}

    def visit(node, reduce=False):
        key = node, reduce
        if key in visited:
            return key
        op, _, args, _ = nodes[node]
        action = op
        if not reduce or op == "matmul" and not shared[node]:
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
        for dependency in dependencies:
            users[dependency] = users.get(dependency, 0) + 1
        visited.add(key)
        order.append((key, action, dependencies))
        return key

    roots = tuple(visit(root, kind == "shared") for root, _, kind in outputs)
    for key in roots:
        users[key] = users.get(key, 0) + 1
    return tuple(order), tuple(shared), tuple(users.items()), roots


def matrix_product(left, right, shape, shared, edges, reduce):
    """Use GEMM for shared matrices and sum outer products before materializing."""
    rows, inner, columns = shape
    left_shared, right_shared = shared
    left_rows, right_rows = (
        (1 if left_shared else edges),
        (1 if right_shared else edges),
    )
    left = left.reshape(left_rows, rows, inner)
    right = right.reshape(right_rows, inner, columns)
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
    indices = index_function(metadata, inputs[0].device)
    if optimized:
        order, shared, users, roots = program_schedule(metadata)
        users = dict(users)
    else:
        order = tuple(
            ((i, False), op, tuple((arg, False) for arg in args))
            for i, (op, _, args, _) in enumerate(nodes)
        )
        shared = (False,) * len(nodes)
        roots = tuple((root, False) for root, _, _ in outputs)
    values = {}
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
        elif op == "constant":
            value = inputs[0].new_full((), data).expand(rows, size)
        elif op == "add":
            value = operands[0] + operands[1]
        elif op == "mul":
            value = operands[0] * operands[1]
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
        elif op == "gather":
            schedule = (
                column_schedule(data, nodes[args[0]][1])
                if optimized and config.TORCH_STATIC_COLUMNS
                else None
            )
            value = (
                channel_gather(operands[0], schedule)
                if schedule is not None
                else operands[0].index_select(1, indices[node])
            )
        elif op == "scatter":
            schedule = (
                column_schedule(data, size)
                if optimized and config.TORCH_STATIC_COLUMNS
                else None
            )
            value = (
                channel_scatter(operands[0], schedule, size)
                if schedule is not None
                else (
                    inputs[0]
                    .new_zeros((rows, size))
                    .index_add(1, indices[node], operands[0])
                )
            )
        elif op == "transpose":
            height, columns = data
            value = (
                operands[0].reshape(rows, height, columns).transpose(1, 2).flatten(1)
            )
        elif op == "matmul":
            if optimized:
                value = matrix_product(
                    *operands, data, tuple(shared[i] for i in args), edges, reduce
                )
            else:
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
            for arg in dependencies:
                users[arg] -= 1
                if users[arg] == 0:
                    del values[arg]
    result = {}
    used = (
        {nodes[key[0]][3][0] for key, action, _ in order if action == "input"}
        if optimized
        else set()
    )
    unused = tuple(value for slot, value in enumerate(inputs) if slot not in used)
    zero = sum(value[..., :0].sum() for value in unused)
    for (root, slot, kind), key in zip(outputs, roots):
        value = values[key] + zero if unused else values[key]
        if kind in ("source", "target"):
            if shared[root]:
                value = value.expand(edges, -1)
            value = value.new_zeros((counts[kind], value.shape[1])).index_add(
                0, source if kind == "source" else target, value
            )
        elif kind == "shared":
            if not optimized:
                value = value.sum(0, keepdim=True)
        elif shared[root]:
            value = value.expand(edges, -1)
        result[slot] = result[slot] + value if slot in result else value
    return [result[slot] for slot in sorted(result)]
