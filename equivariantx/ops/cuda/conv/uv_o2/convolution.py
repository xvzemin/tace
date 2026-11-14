"""Tiled radial weighting, local channel maps, gates and graph reduction."""

from ....common.edge import evaluate as edge_program
from ....common.uv_o2 import prepare, program_inputs


def evaluate(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    edge_index,
    vectors,
    radial_network,
):
    """Evaluate the complete local convolution with CUDA kernels."""
    edge_features, conv_weights, radial_network, _ = prepare(
        module,
        features,
        edge_features,
        conv_weights,
        edge_index,
        vectors,
        radial_network,
    )
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
        False,
    )
    return edge_program(
        metadata,
        inputs,
        *edge_index,
        features.shape[0],
        inline=False,
    )[0]
