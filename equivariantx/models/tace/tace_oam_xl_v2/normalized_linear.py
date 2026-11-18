"""Associate scalar RMS factors with the following irrep matrix operation.

The learned affine gains multiply the corresponding matrix rows. The common
per-node reciprocal RMS is applied after the matrix product. Parameters and
all derivatives remain live; nothing is cached across updates.
"""

import torch


def normalized_linear(linear, normalization, inputs):
    """Apply merged RMS normalization and a linear map to flat mul_ir features."""
    if normalization.center_scalars or normalization.layout != "mul_ir":
        raise ValueError("Associated RMS linear requires mul_ir RMS normalization")
    matrices = linear.weight
    if not isinstance(matrices, torch.nn.ParameterList):
        matrices = matrices.split([a * b for a, b in linear._path_shapes])
    inverse = normalization.inverse_rms(inputs).unsqueeze(-1)
    blocks = [
        part.unflatten(1, (c, d)).transpose(1, 2)
        for part, c, d in zip(
            inputs.split(normalization.widths, dim=1),
            normalization.channels,
            normalization.components,
        )
    ]
    gains = (
        normalization.weight.split(normalization.channels)
        if normalization.weight is not None
        else None
    )
    pieces = [[] for _ in linear.irreps_out]
    for matrix, instruction in zip(matrices, linear.linear.instructions):
        ci, ir = linear.irreps_in[instruction.i_in]
        co, _ = linear.irreps_out[instruction.i_out]
        weight = matrix.reshape(ci, co)
        if gains is not None:
            weight = weight * gains[instruction.i_in][:, None]
        value = blocks[instruction.i_in] @ weight
        value = value * instruction.path_weight * inverse
        pieces[instruction.i_out].append(
            value.transpose(1, 2).reshape(inputs.shape[0], co * ir.dim)
        )
    outputs, bias_offset = [], 0
    for (co, ir), terms in zip(linear.irreps_out, pieces):
        value = terms[0] if terms else inputs.new_zeros((inputs.shape[0], co * ir.dim))
        for term in terms[1:]:
            value = value + term
        if ir.is_scalar() and linear.bias is not None:
            value = value + linear.bias[bias_offset : bias_offset + co]
            bias_offset += co
        outputs.append(value)
    return torch.cat(outputs, -1)
