"""CUDA projection of planar Cartesian tensors."""

import math

import torch

from ...ops.channel_product import channel_product


def projector_metadata(irreps):
    """Return analytic Cartesian STF coefficients in mul_ir layout."""
    rows = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        for i in range(ir.dim):
            for j in range(ir.dim):
                phase = (i.bit_count() - j.bit_count()) % 4
                coefficient = (
                    (1.0, 0.0, -1.0, 0.0)[phase] * 2.0 ** (1 - ir.m) if ir.m else 1.0
                )
                if coefficient:
                    rows.append(
                        (
                            (section.start + i, section.start + j),
                            coefficient,
                            mul,
                            (ir.dim, ir.dim),
                        )
                    )
    return repr((0, (irreps.dim, irreps.dim), tuple(rows)))


def project(metadata, features):
    """Apply the native Cartesian projector and its recursive transpose."""
    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        raise ValueError("The CUDA projector requires float32 or float64 CUDA tensors.")
    dim = features.shape[-1]
    operands = [
        features.reshape(math.prod(features.shape[:-1]), dim),
        features.new_empty((0, dim)),
    ]
    return channel_product(metadata, repr((((0, 1), 1, 0, 0),)), operands)[0].reshape(
        features.shape
    )
