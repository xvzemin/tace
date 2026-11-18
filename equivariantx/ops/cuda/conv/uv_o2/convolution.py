"""Tiled radial weighting, local channel maps, gates and graph reduction."""

from ....common.cuda.channel_linear import scale_parameters
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
    parameters = list(parameters)
    for i, blocks in enumerate(module.path_scales):
        weight = parameters[2 * i]
        factors = getattr(module, f"scale_rows_{i}", None)
        if (
            factors is not None
            and factors.device == weight.device
            and factors.dtype == weight.dtype
        ):
            factors = factors.view(-1, 1)
            parameters[2 * i] = (
                weight.reshape(factors.shape[0], -1) * factors
            ).reshape_as(weight)
        else:
            parameters[2 * i] = scale_parameters(weight, blocks)
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
        normalized=True,
    )
    return edge_program(
        metadata,
        inputs,
        *edge_index,
        features.shape[0],
        inline=False,
    )[0]
