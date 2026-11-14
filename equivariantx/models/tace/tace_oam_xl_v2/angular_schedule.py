"""Schedule scalar CG temporaries close to their final derivative consumers."""

import re
from collections import Counter


def schedule_mode(mode, program):
    """Retain the original force sweep; select measured forward/reverse orders."""
    if mode != "auto":
        return mode
    roles = {role for _, _, pairs in program for role, _ in pairs}
    return "greedy" if roles == {4} else "nodes" if roles == {0, 1} else "off"


def stream_body(body, sinks, mode):
    """Topologically emit scalar expressions, retaining each expression's order.

    Parameters
    ----------
    body : list of str
        Straight-line scalar declarations and updates from CG code generation.
    sinks : list of (list of str, int)
        Output-reduction blocks and their derivative roles. No output aliases
        an input. Each block's original summation order is preserved.
    mode : {'nodes', 'weights', 'greedy', 'live', 'peak'}
        Order independent output blocks to shorten temporary live ranges.
    """
    groups = {}
    for line in body:
        match = re.match(r"\s*(?:(?:const )?T )?(v\d+)\s*=", line)
        if match is None:
            return body + [line for block, _ in sinks for line in block]
        groups.setdefault(match[1], []).append(line)
    dependencies = {
        name: list(
            dict.fromkeys(
                v
                for line in lines
                for v in re.findall(r"\bv\d+\b", line)
                if v != name and v in groups
            )
        )
        for name, lines in groups.items()
    }
    roots = [
        list(
            dict.fromkeys(
                v
                for line in block
                for v in re.findall(r"\bv\d+\b", line)
                if v in groups
            )
        )
        for block, _ in sinks
    ]
    closures = []
    for root in roots:
        needed = set()
        stack = list(root)
        while stack:
            name = stack.pop()
            if name not in needed:
                needed.add(name)
                stack.extend(dependencies[name])
        closures.append(needed)
    remaining = Counter(name for closure in closures for name in closure)
    live = set()
    result, emitted, visiting = [], set(), set()

    def emit(name):
        if name in emitted:
            return
        if name in visiting:
            raise ValueError("Scalar CG dependency cycle")
        visiting.add(name)
        for parent in dependencies[name]:
            emit(parent)
        result.extend(groups[name])
        visiting.remove(name)
        emitted.add(name)

    pending = list(range(len(sinks)))
    if mode == "weights":
        pending.sort(key=lambda i: sinks[i][1] != 1)
    while pending:
        if mode in ("live", "peak"):

            def pressure(i):
                closure = closures[i]
                new_count = len(closure - emitted)
                after = len(live - closure) + sum(
                    remaining[name] > 1 for name in closure
                )
                peak = len(live) + new_count
                return (
                    (after, peak, new_count)
                    if mode == "live"
                    else (peak, after, new_count)
                )

            index = min(pending, key=pressure)
            pending.remove(index)
        elif mode == "greedy":

            def missing(i):
                seen = set()
                stack = list(roots[i])
                while stack:
                    node = stack.pop()
                    if node in seen or node in emitted:
                        continue
                    seen.add(node)
                    stack.extend(dependencies[node])
                return len(seen)

            index = min(pending, key=missing)
            pending.remove(index)
        else:
            index = pending.pop(0)
        for root in roots[index]:
            emit(root)
        result.extend(sinks[index][0])
        live.update(closures[index])
        for name in closures[index]:
            remaining[name] -= 1
            if remaining[name] == 0:
                live.discard(name)
    return result
