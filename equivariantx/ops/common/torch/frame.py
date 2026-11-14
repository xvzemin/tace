"""Indexed, degree-preserving and rotary frame contractions."""

import torch

from ....utils.metadata import parse_metadata
from ....utils.program import Program
from .edge import evaluate_torch


def indexed_frame(nodes, matrix, edges, index, num_nodes, layout, role):
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
        metadata, (nodes, matrix, edges), index, index, num_nodes, optimized=True
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
