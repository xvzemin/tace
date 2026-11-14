"""Grouped frame rotations and explicit gather/scatter operations."""

import torch

from ....utils.layout import permute


def to_local(module, features, wigner):
    """Rotate validated blocks using grouped batched matrix products."""
    matrices = module._rotation_matrices(wigner.to(features))
    if not module._global_identity:
        features = permute(features, module._global_index, module._global_inverse)
    outputs = []
    for degree, mul, matrix, values in zip(
        module._degrees,
        module._multiplicities,
        matrices,
        features.split(module._global_sizes, dim=-1),
    ):
        values = values.view(*features.shape[:-1], 2 * degree + 1, mul)
        values = (
            torch.bmm(matrix, values)
            if features.ndim == 2
            else torch.einsum("bij,b...jk->b...ik", matrix, values)
        )
        outputs.append(values.flatten(-2))
    if not outputs:
        return features[..., :0]
    features = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
    if not module._local_identity:
        features = permute(features, module._local_index, module._local_inverse)
    return features * module._local_sign.to(features) if module._has_signs else features


def to_global(module, features, wigner_inv):
    """Rotate validated blocks using grouped batched matrix products."""
    matrices = module._rotation_matrices(wigner_inv.to(features), inverse=True)
    if module._has_signs:
        features = features * module._local_sign.to(features)
    if not module._local_identity:
        features = permute(features, module._local_inverse, module._local_index)
    outputs = []
    for degree, mul, matrix, values in zip(
        module._degrees,
        module._multiplicities,
        matrices,
        features.split(module._local_sizes, dim=-1),
    ):
        retained = 2 * min(degree, module.mmax) + 1
        values = values.view(*features.shape[:-1], retained, mul)
        values = (
            torch.bmm(matrix, values)
            if features.ndim == 2
            else torch.einsum("bij,b...jk->b...ik", matrix, values)
        )
        outputs.append(values.flatten(-2))
    if not outputs:
        return features[..., :0]
    features = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
    return (
        features
        if module._global_identity
        else permute(features, module._global_inverse, module._global_index)
    )


def gather(module, features, wigner, index):
    """Gather or reduce the ordinary frame transform."""
    return module.to_local(features.index_select(0, index), wigner)


def scatter(module, features, wigner_inv, index, num_nodes):
    """Gather or reduce the ordinary frame transform."""
    messages = module.to_global(features, wigner_inv)
    return messages.new_zeros((num_nodes, module.input_dim)).index_add(
        0, index, messages
    )
