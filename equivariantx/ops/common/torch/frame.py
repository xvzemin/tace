"""Indexed, degree-preserving and rotary frame contractions."""

import torch

from ....utils.metadata import parse_metadata
from ....utils.program import Program
from .edge import evaluate_torch


def indexed_frame(nodes, matrix, edges, index, num_nodes, layout, role):
    return frame_program(
        nodes, matrix, edges, index, num_nodes, layout, role, optimized=True
    )


def indexed_frame_reference(nodes, matrix, edges, index, num_nodes, layout, role):
    return frame_program(
        nodes, matrix, edges, index, num_nodes, layout, role, optimized=False
    )


def frame_program(nodes, matrix, edges, index, num_nodes, layout, role, *, optimized):
    """Evaluate a local-frame pairing with the declared node reduction."""
    ni, no, nw, _ = parse_metadata(layout)
    dimensions = ni, nw, no
    program = Program()
    operands = tuple(
        program.constant(width, 0)
        if slot == role
        else program.input(slot, "source" if slot == 0 else "edge", width)
        for slot, width in enumerate(dimensions)
    )
    result = program.add("frame", dimensions[role], operands, (layout, role))
    metadata = repr(
        (
            tuple(program.nodes),
            ((result, 0, "source" if role == 0 else "edge"),),
        )
    )
    return evaluate_torch(
        metadata, (nodes, matrix, edges), index, index, num_nodes, optimized=optimized
    )[0]


def frame_rotation(features, matrix, edge, metadata, role):
    """Apply matrices only between angular coordinates of equal degree."""
    degrees_in, degrees_out = parse_metadata(metadata)
    mask = matrix.new_tensor(
        [[left == right for right in degrees_in] for left in degrees_out]
    )
    if role == 1:
        result = torch.bmm(edge, features.transpose(1, 2)) * mask
    elif role == 0:
        result = torch.bmm((matrix * mask).transpose(1, 2), edge)
    else:
        result = torch.bmm(matrix * mask, features)
    return result + (features, matrix, edge)[role][..., :0].sum()


def frame_reference(features, matrix, edge, metadata, role):
    degrees_in, degrees_out = parse_metadata(metadata)
    mask = matrix.new_tensor([[a == b for a in degrees_in] for b in degrees_out])
    terms = [features, matrix, edge]
    labels = ["eic", "eoi", "eoc"]
    output = labels.pop(role)
    dummy = terms.pop(role)
    result = torch.einsum(",".join((*labels, "oi")) + "->" + output, *terms, mask)
    return result + dummy[..., :0].sum()


def rotary_product(query, key, phase, score, lmax, mmax, heads, role):
    """Evaluate the head-wise rotary pairing or one multilinear transpose."""
    rows, angular, channels = query.shape
    if channels % heads:
        raise ValueError("The channel width must be divisible by the head count.")
    query, key = [
        value.reshape(rows, angular, heads, channels // heads) for value in (query, key)
    ]
    phases = phase.reshape(rows, mmax + 1, heads, 2)
    parts, start = [], 0
    for order in range(mmax + 1):
        width = lmax + 1 - order
        qr, kr = [value[:, start : start + width] for value in (query, key)]
        if order:
            qi, ki = [
                value[:, start + width : start + 2 * width] for value in (query, key)
            ]
        real = phases[:, order, :, 0][:, None, :, None]
        imag = phases[:, order, :, 1][:, None, :, None]
        if role in (0, 1):
            vr = kr if role == 0 else qr
            vr = real * vr
            if order:
                vi = ki if role == 0 else qi
                vr = vr + (-1 if role == 0 else 1) * imag * vi
                vi = real * vi + (1 if role == 0 else -1) * imag * (
                    kr if role == 0 else qr
                )
                vr = torch.cat((vr, vi), dim=1)
            parts.append(vr * score[:, None, :, None])
        else:
            real_dot = (qr * kr).sum((1, 3))
            if order:
                real_dot = real_dot + (qi * ki).sum((1, 3))
                imag_dot = (qi * kr - qr * ki).sum((1, 3))
            else:
                imag_dot = torch.zeros_like(real_dot)
            if role == 2:
                parts.append(
                    torch.stack((real_dot, imag_dot), dim=-1) * score[..., None]
                )
            else:
                parts.append(
                    real_dot * phases[:, order, :, 0]
                    + imag_dot * phases[:, order, :, 1]
                )
        start += width * (2 if order else 1)
    if role < 2:
        result = torch.cat(parts, dim=1).reshape(rows, angular, channels)
    elif role == 2:
        result = torch.stack(parts, dim=1).reshape(phase.shape)
    else:
        result = sum(parts[1:], parts[0])
    return result + (query, key, phase, score)[role][..., :0].sum()


def rotary_reference(query, key, phase, score, lmax, mmax, heads, role):
    """Differentiate the scalar pairing to select any multilinear output leg."""

    def primal(q, k, p):
        q, k = (v.unflatten(-1, (heads, -1)) for v in (q, k))
        p = p.reshape(q.shape[0], mmax + 1, heads, 2)
        offset, parts = 0, []
        for m in range(mmax + 1):
            width = lmax + 1 - m
            qr, kr = (v[:, offset : offset + width] for v in (q, k))
            alpha = p[:, m, :, 0][:, None, :, None]
            block = qr * kr * alpha
            if m:
                qi, ki = (v[:, offset + width : offset + 2 * width] for v in (q, k))
                beta = p[:, m, :, 1][:, None, :, None]
                block = block + qi * ki * alpha + (qi * kr - qr * ki) * beta
            parts.append(block.sum((1, 3)))
            offset += width * (1 + (m != 0))
        return sum(parts[1:], parts[0])

    values = query, key, phase, score
    result = (
        primal(query, key, phase)
        if role == 3
        else torch.func.grad(
            lambda q, k, p, s: (primal(q, k, p) * s).sum(), argnums=role
        )(*values)
    )
    return result + values[role][..., :0].sum()
