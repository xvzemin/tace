"""Fused indexed frame rotations without global edge intermediates."""


def gather(module, features, wigner, index):
    """Fuse indexed frame transformation and node reduction."""
    from equivariantx.ops.common.frame import transform

    return transform(module, features, wigner, index, features.size(0), False)


def scatter(module, features, wigner_inv, index, num_nodes):
    """Fuse indexed frame transformation and node reduction."""
    from equivariantx.ops.common.frame import transform

    return transform(module, features, wigner_inv, index, num_nodes, True)
