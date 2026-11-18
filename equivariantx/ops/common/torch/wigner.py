"""Ordinary Torch evaluation of native frame and recursive rotation programs."""

import torch

from ....utils.metadata import parse_metadata
from ....utils.scalar import alignment_program


def alignment_reference(key, inputs):
    """Evaluate the same scalar geometry expressions at every transpose order."""
    nodes, outputs, _ = alignment_program(
        parse_metadata(key), "float" if inputs[0].dtype == torch.float32 else "double"
    )
    needed = {node for group in outputs for node in group}
    for i in reversed(range(len(nodes))):
        if i in needed and nodes[i][0] not in ("input", "constant"):
            needed.update(nodes[i][1:])
    values = {}
    for i, (op, *args) in enumerate(nodes):
        if i not in needed:
            continue
        if op == "input":
            value = inputs[args[0]][:, args[1]]
        elif op == "constant":
            value = inputs[0].new_full((inputs[0].shape[0],), args[0])
        else:
            operands = [values[arg] for arg in args]
            if op == "select":
                value = torch.where(*operands)
            else:
                value = getattr(torch, op)(*operands)
        values[i] = value
    zero = sum(value[..., :0].sum() for value in inputs)
    return [
        torch.stack([values[node] for node in group], dim=-1) + zero
        for group in outputs
    ]


def alignment(key, inputs):
    tag = parse_metadata(key)
    if tag == "normalize":
        x = inputs[0]
        return [x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)]
    if tag == "direction_gradient":
        x, torque = inputs
        return [torch.linalg.cross(torque, x) / x.square().sum(-1, keepdim=True)]
    if tag is None or tag == "quaternion":
        from ...torch.o2.rotation_matrix import reference

        x = inputs[0]
        return [reference(x, 1, 1e-7, tag == "quaternion").flatten(1)]
    return alignment_reference(key, inputs)


def rotation_reference(cg, output, values):
    """Contract one recursive Wigner degree or any of its bilinear transposes."""
    widths = 3, cg.shape[1], cg.shape[2]
    tokens = "eac", "ebd", "emn"
    operands, labels = [], []
    for role, (value, width) in enumerate(zip(values, widths)):
        if role != output:
            operands.append(value.reshape(-1, width, width))
            labels.append(tokens[role])
    expression = ",".join(("abm", "cdn", *labels)) + "->" + tokens[output]
    result = torch.einsum(expression, cg, cg, *operands) * cg.shape[2]
    return result.flatten(1) + values[output][..., :0].sum()


def rotation(cg, output, values):
    if output != 2:
        return rotation_reference(cg, output, values)
    d1, d2 = cg.shape[1:]
    a = values[0].reshape(-1, 3, 3)
    b = values[1].reshape(-1, d1, d1)
    t = torch.einsum("ebd,cdn->ebcn", b, cg)
    t = torch.einsum("eac,ebcn->eabn", a, t)
    return (torch.einsum("abm,eabn->emn", cg, t) * d2).flatten(1) + values[2][
        ..., :0
    ].sum()


def packed_wigner(vectors, degree, coefficients):
    return build_wigner(vectors, degree, coefficients, alignment, rotation)


def packed_reference(vectors, degree, coefficients):
    return build_wigner(
        vectors, degree, coefficients, alignment_reference, rotation_reference
    )


def build_wigner(vectors, degree, coefficients, align, rotate):
    """Build every degree with differentiable geometry and ordinary contractions."""
    matrices = [vectors.new_ones((vectors.shape[0], 1))]
    if degree:
        matrices.append(align("None", [vectors])[0])
    for cg in coefficients:
        placeholder = vectors.new_empty((vectors.shape[0], cg.shape[2] ** 2))
        matrices.append(rotate(cg, 2, [matrices[1], matrices[-1], placeholder]))
    return torch.cat(matrices, dim=-1) + vectors[..., :0].sum()
