"""Native channelwise direction contractions."""

from ....torch.conv.uu_o2 import convolution as torch_kernel


def transform_weights(module, weights):
    """Fuse the transverse weight transformation."""
    from equivariantx.ops.common.channel_product import local_product

    return local_product(module.weight_metadata, weights)


def wigner(
    module,
    features,
    radial,
    projection,
    wigner,
    cutoff,
    edge_index,
    num_nodes,
    vectors,
    radial_network,
):
    """Contract packed directions and their analytic generator derivatives."""
    if vectors is None or vectors.dtype != features.dtype or radial_network is not None:
        return torch_kernel.wigner(
            module,
            features,
            radial,
            projection,
            wigner,
            cutoff,
            edge_index,
            num_nodes,
            vectors,
            radial_network,
        )
    from equivariantx.ops.common.geometry import direction_contraction

    operands = [
        features,
        radial,
        projection,
        wigner.detach(),
        cutoff,
        features.new_empty(1).expand(num_nodes, module.output_dim),
    ]
    return direction_contraction(
        module.direction_metadata,
        repr(((0, (0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),)),
        vectors,
        edge_index[0],
        edge_index[1],
        operands,
        module.backend == "cuda",
    )[0]
