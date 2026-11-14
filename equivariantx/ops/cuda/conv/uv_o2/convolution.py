"""Native local convolutions with shared GEMMs for attention."""

from ....common.edge import evaluate as edge_program
from ....torch.conv.uv_o2.convolution import prepare, program_inputs, staged


def evaluate(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    radial_attention,
    edge_index,
    vectors,
    radial_network,
):
    """Fuse non-attention messages or retain GEMMs around native reductions."""
    edge_features, conv_weights, radial_network, expanded = prepare(
        module,
        features,
        edge_features,
        conv_weights,
        edge_index,
        vectors,
        radial_network,
    )
    if not module.attention:
        metadata, inputs = program_inputs(
            module,
            features,
            edge_features,
            conv_weights,
            wigner,
            wigner_inv,
            cutoff,
            parameters,
            *edge_index,
            vectors,
            radial_network,
            expanded,
        )
        if expanded:
            from equivariantx.ops.common.torch.edge import evaluate_torch

            return evaluate_torch(
                metadata, inputs, *edge_index, features.shape[0], optimized=True
            )[0]
        return edge_program(metadata, inputs, *edge_index, features.shape[0])[0]
    return staged(
        module,
        features,
        edge_features,
        conv_weights,
        wigner,
        wigner_inv,
        cutoff,
        parameters,
        radial_attention,
        edge_index,
        vectors,
        radial_network,
        edge_program,
        native=True,
    )
