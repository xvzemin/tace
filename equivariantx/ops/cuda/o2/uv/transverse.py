"""Indexed restriction and lifting in spherical transverse storage."""


def evaluate(module, features, direction, source, target, num_nodes, inverse):
    """Restrict or lift validated features and aggregate matching edges."""
    if inverse:
        features = features.index_select(1, module.inverse_permutation)
    width = module.input_dim if inverse else module.output_dim
    from equivariantx.ops.common.o3 import contraction

    values = [
        features,
        features.new_empty((1, 0)),
        features.new_empty((0, 0)),
        features.new_ones((1, 1)),
        features.new_empty(1).expand(num_nodes, width),
        direction,
    ]
    result = contraction(
        module.metadata[inverse],
        repr(((tuple(range(6)), False, ((4, 0),)),)),
        source,
        target,
        values,
    )[0]
    return result if inverse else result.index_select(1, module.permutation)
