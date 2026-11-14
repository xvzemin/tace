"""Pruned scalar expression schedules shared by Torch and CUDA."""

from functools import lru_cache

import torch

from ...utils.scalar import ScalarProgram, alignment_program


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def rotation_program(axis, eps, quaternion, dtype, derivative=()):
    """Build an alignment chart or its recursively transposed derivatives."""
    if derivative:
        parent, active = derivative
        nodes, outputs, widths = rotation_program(axis, eps, quaternion, dtype, parent)
        program = ScalarProgram(nodes)
        for i, width in enumerate(widths):
            for j in range(width):
                program.add("input", i, j)
        outputs, widths = program.transpose(outputs, widths, active)
        nodes = tuple(program.nodes)
    else:
        nodes, outputs, widths = alignment_program(
            "quaternion" if quaternion else None, dtype
        )
        permutation = ((2, 0, 1), (0, 1, 2), (1, 2, 0))[axis]
        nodes = tuple(
            ("input", 0, permutation[node[2]])
            if node[0] == "input"
            else ("constant", eps * eps)
            if node == ("constant", 1e-14)
            else node
            for node in nodes
        )
        inverse = tuple(permutation.index(k) for k in range(3))
        if quaternion:
            outputs = ((outputs[0][0], *(outputs[0][1 + k] for k in inverse)),)
        else:
            outputs = (tuple(outputs[0][3 * i + j] for i in inverse for j in inverse),)
    return prune(nodes, outputs, widths)


def prune(nodes, outputs, widths):
    """Retain only reachable expressions without changing input slots."""
    needed = {node for group in outputs for node in group}
    for index in range(len(nodes) - 1, -1, -1):
        if index in needed and nodes[index][0] not in ("input", "constant"):
            needed.update(nodes[index][1:])
    locations = {}
    result = []
    for index, (op, *arguments) in enumerate(nodes):
        if index not in needed:
            continue
        locations[index] = len(result)
        if op not in ("input", "constant"):
            arguments = tuple(locations[k] for k in arguments)
        result.append((op, *arguments))
    return (
        tuple(result),
        tuple(tuple(locations[k] for k in group) for group in outputs),
        widths,
    )
