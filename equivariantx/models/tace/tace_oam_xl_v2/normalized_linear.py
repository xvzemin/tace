"""Associate scalar RMS factors with the following irrep matrix operation.

The learned affine gains multiply the corresponding matrix rows. The common
per-node reciprocal RMS is applied after the matrix product. Parameters and
all derivatives remain live; nothing is cached across updates.
"""

import torch


def normalized_linear(linear, normalization, reshape, inputs):
    """Evaluate a non-centering equivariant RMSNorm followed by a linear map."""
    if normalization.centering or normalization.affine_bias is not None:
        raise ValueError("Associated RMS linear requires no centering or affine bias")
    if not isinstance(linear.weight, torch.nn.ParameterList):
        raise ValueError(
            "Associated RMS linear requires independent parameter matrices"
        )
    shaped = reshape(inputs)
    norm = shaped.pow(2).mean(dim=2, keepdim=True)
    if normalization.normalization == "norm":
        norm = norm.sum(dim=1, keepdim=True)
    elif normalization.std_balance_degrees:
        norm = torch.einsum("ai,nic->nac", normalization.balance_degree_weight, norm)
    else:
        norm = norm.mean(dim=1, keepdim=True)
    inverse = (norm + normalization.eps).pow(-0.5)
    offsets, offset = [], 0
    for ci, ir in linear.irreps_in:
        if ci != normalization.num_channels:
            raise ValueError("Input irreps must match normalization channels")
        offsets.append(offset)
        offset += ir.dim
    pieces = [[] for _ in linear.irreps_out]
    for matrix, instruction in zip(linear.weight, linear.linear.instructions):
        ci, ir = linear.irreps_in[instruction.i_in]
        co, _ = linear.irreps_out[instruction.i_out]
        weight = matrix.reshape(ci, co)
        if normalization.affine_weight is not None:
            weight = weight * normalization.affine_weight[instruction.i_in, :, None]
        value = (
            shaped[:, offsets[instruction.i_in] : offsets[instruction.i_in] + ir.dim]
            @ weight
        )
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
