"""Spherical CG convolution in angular-first storage."""

from ....common.o3 import contraction


def evaluate(
    module,
    features,
    edge_attrs,
    radial,
    projection,
    edge_index,
    num_nodes,
    vectors,
    amplitudes,
    radial_network,
):
    """Evaluate validated convolution operands."""
    indexed = edge_index is not None
    operands = [
        features,
        radial,
        projection,
        edge_attrs,
        features.new_empty(1).expand(
            num_nodes,
            module.harmonic_output_dim if vectors is not None else module.output_dim,
        ),
    ]
    if vectors is not None:
        operands.append(vectors)
    if radial_network is not None:
        from equivariantx.ops.common.radial import convolve

        return convolve(
            "o3",
            module.harmonic_metadata if vectors is not None else module.kernel_metadata,
            operands,
            edge_index,
            radial_network,
        )
    return contraction(
        module.harmonic_metadata if vectors is not None else module.kernel_metadata,
        repr(((tuple(range(len(operands))), False, ((4, 0),)),)),
        edge_index[0] if indexed else None,
        edge_index[1] if indexed else None,
        operands,
        batch_paths=module.batch_paths,
    )[0]
