"""Compile a gated CG product as one differentiable multilinear operation."""

import torch

from equivariantx.ops.channel_product import channel_product
from equivariantx.utils.metadata import channel_product_layout


def gated_product_metadata(metadata, gate_index, gate_width):
    """Insert a shared per-path gate into every retained CG monomial.

    Parameters
    ----------
    metadata : str
        EQX channel-product metadata in flattened channel/irrep layout.
    gate_index : sequence of int
        The original mapping from expanded output columns to gate columns.
    gate_width : int
        Width of the compact gate input.

    Returns
    -------
    str
        Four-operand product contract, including the result role.
    """
    dimensions, rows = channel_product_layout(metadata)
    if len(dimensions) != 3 or len(gate_index) != dimensions[2]:
        raise ValueError(
            "Expected an ordinary bilinear product and its full gate mapping"
        )
    combined = []
    for indices, coefficient, width, strides in rows:
        if any(index < 0 for index in indices):
            raise ValueError("Gated CG products require all bilinear operands")
        output, step = indices[2], strides[2]
        gate_start = int(gate_index[output])
        gate_step = int(gate_index[output + step]) - gate_start if width > 1 else 0
        if any(
            int(gate_index[output + c * step]) != gate_start + c * gate_step
            for c in range(width)
        ):
            raise ValueError("Gate mapping is not affine along product channels")
        if (
            not 0 <= gate_start < gate_width
            or not 0 <= gate_start + (width - 1) * gate_step < gate_width
        ):
            raise ValueError("Gate mapping exceeds the compact gate width")
        combined.append(
            (
                (indices[0], indices[1], gate_start, output),
                coefficient,
                width,
                (strides[0], strides[1], gate_step, step),
            )
        )
    return repr(
        (0, (dimensions[0], dimensions[1], gate_width, dimensions[2]), tuple(combined))
    )


def gated_product(metadata, left, right, gate, output_width):
    """Evaluate the product and all requested mixed adjoints with EQX kernels."""
    empty = left.new_empty((0, output_width))
    return channel_product(
        metadata, repr((((0, 1, 2, 3), 3, 0, 0),)), [left, right, gate, empty]
    )[0]
