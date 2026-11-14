"""Block basis transforms and planar symmetric-traceless projection."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config


def change_basis(module, features):
    """Apply equal-order blocks using dense or factored basis matrices."""
    values = []
    for mul, m, dim, circular_dim, section in module.paths:
        matrix = getattr(module, f"basis_{m}")
        x = features[..., section]
        if module.inverse:
            x = x.reshape(*features.shape[:-1], mul, dim) @ matrix
            x = x.transpose(-1, -2)
        else:
            x = x.reshape(*features.shape[:-1], circular_dim, mul)
            x = x.transpose(-1, -2) @ matrix.T
        values.append(x.flatten(-2))
    return torch.cat(values, -1) if values else features[..., :0]


def project_blocks(module, features, transform):
    """Apply one transform to each group of equal-order Cartesian blocks."""
    values = [
        features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
        for (mul, ir), section in zip(module.irreps_in, module.irreps_in.slices())
    ]
    for m, entries in module.groups:
        value = (
            values[entries[0][0]]
            if len(entries) == 1
            else torch.cat([values[i] for i, _ in entries], dim=-2)
        )
        value = transform(m, value)
        for (i, _), part in zip(entries, value.split([mul for _, mul in entries], -2)):
            values[i] = part
    return (
        torch.cat([x.flatten(-2) for x in values], -1) if values else features[..., :0]
    )


def project_matrix(module, rank, features):
    """Apply the explicitly selected dense or factored reference matrix."""
    matrix = getattr(module, f"matrix_{rank}")
    result = features @ matrix
    return result @ matrix.T if module.method == "factorized" else result


def project_reference(module, features):
    """Project using independent fixed matrices and ordinary autograd."""
    return project_blocks(
        module, features, lambda rank, value: project_matrix(module, rank, value)
    )


def planar_project(features, rank):
    """Contract two signed axes instead of a square planar STF matrix."""
    if rank < 2:
        return features.clone()
    phases = tuple(index.bit_count() % 4 for index in range(2**rank))
    real = features.new_tensor([(1.0, 0.0, -1.0, 0.0)[i] for i in phases])
    imag = features.new_tensor([(0.0, 1.0, 0.0, -1.0)[i] for i in phases])
    a = (features * real).sum(-1, keepdim=True)
    b = (features * imag).sum(-1, keepdim=True)
    return (a * real + b * imag) * 2.0 ** (1 - rank)


def project(module, features):
    """Expose bounded signed reductions to the joint Torch compiler."""
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_PLANAR_PROJECTION
        or module.method != "matrix"
        or not (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return project_reference(module, features)

    def transform(rank, value):
        if 4 <= rank <= 8:
            return planar_project(value, rank)
        return project_matrix(module, rank, value)

    return project_blocks(module, features, transform)
