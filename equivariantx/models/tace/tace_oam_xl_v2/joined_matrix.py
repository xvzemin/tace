"""Join compatible live GEMMs in the complete explicit gradient graph.

For a common left operand, ``[A B1, A B2] = A [B1, B2]``; for a common
right operand, stack rows instead. Original parameters and their optimizer
groups are retained. This pass runs after differentiation and supports both
forward matrices and mixed parameter gradients without frozen weight caches.
"""

from collections import defaultdict
import heapq
import operator

import torch


def reorder_graph(graph):
    """Restore a stable topological order after independent products join."""
    nodes = list(graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    degree = {node: len(node.all_input_nodes) for node in nodes}
    ready = [order[node] for node in nodes if degree[node] == 0]
    heapq.heapify(ready)
    sorted_nodes = []
    while ready:
        node = nodes[heapq.heappop(ready)]
        sorted_nodes.append(node)
        for user in node.users:
            degree[user] -= 1
            if degree[user] == 0:
                heapq.heappush(ready, order[user])
    if len(sorted_nodes) != len(nodes):
        raise RuntimeError("Joining matrices introduced a dependency cycle")
    cursor = sorted_nodes[0]
    for node in sorted_nodes[1:]:
        cursor.append(node)
        cursor = node
    graph.lint()


def independent(nodes):
    """All member operands must be independent of every member output."""
    members = set(nodes)
    pending = [arg for node in nodes for arg in node.all_input_nodes]
    seen = set()
    while pending:
        node = pending.pop()
        if node in members:
            return False
        if node not in seen:
            seen.add(node)
            pending.extend(node.all_input_nodes)
    return True


def join_matrix_graph(graph, maximum_span=192, maximum_group=4):
    """Join nearby GEMMs sharing a literal live operand, with cycle checks."""
    from .program import refresh_inputs

    counts = dict(left_groups=0, right_groups=0, joined_matrices=0)

    def call(target, args):
        node = graph.call_function(target, args)
        node.meta["val"] = target(*torch.fx.map_arg(args, lambda n: n.meta["val"]))
        refresh_inputs(node)
        return node

    for side in (0, 1):
        nodes = list(graph.nodes)
        position = {node: i for i, node in enumerate(nodes)}
        groups = defaultdict(list)
        for node in nodes:
            if node.target == torch.ops.aten.mm.default and node.meta["val"].dtype in {
                torch.float32,
                torch.float64,
            }:
                groups[node.args[side]].append(node)
        used = set()
        for shared, members in groups.items():
            for first in members:
                if first in used or first.args[side] is not shared:
                    continue
                selected = [first]
                for candidate in members:
                    if (
                        candidate in used
                        or candidate is first
                        or candidate.args[side] is not shared
                    ):
                        continue
                    if not (0 <= position[candidate] - position[first] <= maximum_span):
                        continue
                    if independent([*selected, candidate]):
                        selected.append(candidate)
                    if len(selected) == maximum_group:
                        break
                if len(selected) < 2:
                    continue
                # Inputs can become ready later than the first original GEMM.
                # The dependency check and stable sort preserve all consumers.
                with graph.inserting_before(first):
                    other = [node.args[1 - side] for node in selected]
                    dimension = 1 - side
                    sizes = [node.meta["val"].shape[dimension] for node in selected]
                    joined = call(torch.ops.aten.cat.default, (other, dimension))
                    value = call(
                        torch.ops.aten.mm.default,
                        (shared, joined) if side == 0 else (joined, shared),
                    )
                    parts = call(
                        torch.ops.aten.split_with_sizes.default,
                        (value, sizes, dimension),
                    )
                    for i, node in enumerate(selected):
                        part = call(operator.getitem, (parts, i))
                        node.replace_all_uses_with(part)
                used.update(selected)
                for node in selected:
                    graph.erase_node(node)
                counts["left_groups" if side == 0 else "right_groups"] += 1
                counts["joined_matrices"] += len(selected)
        reorder_graph(graph)
    graph.eliminate_dead_code()
    graph.lint()
    return counts
