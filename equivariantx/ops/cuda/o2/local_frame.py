"""Fused indexed frame rotations without global edge intermediates."""

from math import prod

from ...torch.o2 import local_frame as layout


def multiply(matrix, values):
    from ...common.cuda.matmul import matmul

    if values.ndim == 3:
        return matmul(matrix, values)
    # Rotate a single wide RHS per frame, not a copy of D for each sample.
    rows = values.movedim(-2, 1).reshape(
        values.shape[0], values.shape[-2], prod(values.shape[1:-2]) * values.shape[-1]
    )
    return (
        matmul(matrix, rows)
        .reshape(
            values.shape[0], matrix.shape[-2], *values.shape[1:-2], values.shape[-1]
        )
        .movedim(1, -2)
    )


def permute(features, index, inverse):
    return features.index_select(-1, index)


def to_local(module, features, wigner):
    return layout.to_local(
        module, features, wigner, product=multiply, permutation=permute
    )


def to_global(module, features, wigner):
    return layout.to_global(
        module, features, wigner, product=multiply, permutation=permute
    )


def gather(module, features, wigner, index):
    """Fuse indexed frame transformation and node reduction."""
    from equivariantx.ops.common.frame import transform

    return transform(module, features, wigner, index, features.size(0), False)


def scatter(module, features, wigner_inv, index, num_nodes):
    """Fuse indexed frame transformation and node reduction."""
    from equivariantx.ops.common.frame import transform

    return transform(module, features, wigner_inv, index, num_nodes, True)
