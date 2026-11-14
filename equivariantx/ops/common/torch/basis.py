"""Shared block layout and dense basis-matrix operations."""

import torch


def project_features(module, features, kernel):
    """Apply a block projector to flattened Cartesian features."""
    if isinstance(module, torch.nn.Identity) or not module.groups:
        return features
    values = [
        features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
        for (mul, ir), section in zip(module.irreps_in, module.slices)
    ]
    values = [value.flatten(-2) for value in kernel(module, values)]
    return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


def map_blocks(features, groups, function):
    """Batch equal-rank entries before applying a linear projection."""
    values = list(features)
    for degree, entries in groups:
        value = (
            values[entries[0][0]]
            if len(entries) == 1
            else torch.cat([values[index] for index, _ in entries], -2)
        )
        value = function(degree, value)
        for (index, _), part in zip(
            entries, value.split([mul for _, mul in entries], -2)
        ):
            values[index] = part
    return values


def project_dense(features, matrix, *, square=False, symmetric_basis=None):
    """Apply a dense projector or a factored orthonormal embedding."""
    if symmetric_basis is not None:
        features = symmetric_basis.pack(features)
    value = features @ matrix
    if not square:
        value = value @ matrix.T
    return value if symmetric_basis is None else symmetric_basis.unpack(value)


def transform_tensor(tensor, matrix, rank):
    """Apply a rectangular matrix to every index of a flattened tensor."""
    a, b = matrix.shape[-2:]
    for k in range(rank):
        tensor = tensor.reshape(*tensor.shape[:-1], b ** (rank - k - 1), b, a**k)
        tensor = torch.einsum("...uiv,...ji->...ujv", tensor, matrix).flatten(-3)
    return tensor
