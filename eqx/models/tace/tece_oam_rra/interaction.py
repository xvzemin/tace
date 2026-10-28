"""CUDA local updates with receiver-normalized rotary attention."""

import torch

from ....ops.tece import interaction
from ....ops.tece_program import build
from ....utils.metadata import parse_metadata


def stream(
    module,
    features,
    radial,
    projection,
    bias,
    edge_index,
    cutoff,
    wigner,
    wigner_inv,
    radial_basis,
    *,
    radial_network=None,
):
    """Fuse the complete edge update without graph replay or Python callbacks.

    Edge tiles use matrix products for dense contractions and CUDA kernels
    for intervening expressions. Only attention scores are retained per edge.
    Other intermediates are recomputed in bounded tiles. CUDA frame arguments
    may contain dense matrices or packed degree blocks shared by both directions.
    """
    if bias is None:
        bias = projection.new_zeros(projection.shape[1])
    if cutoff is None:
        cutoff = features.new_ones((edge_index.shape[1], 1))
    if not features.is_cuda or module._eqx_metadata is None:
        if radial_network is not None:
            from ....ops.radial import materialize

            radial = radial_network(materialize(radial, edge_index))
        return module(
            features,
            radial @ projection + bias,
            edge_index,
            cutoff,
            wigner,
            wigner_inv,
            radial_basis,
        )
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("The fused local interaction requires float32 or float64.")
    inputs = [
        module.reshape_in(features),
        radial,
        projection,
        bias,
        cutoff,
        wigner,
        wigner_inv,
        radial_basis,
        *module.parameters(),
    ]
    metadata = module._eqx_metadata
    if radial_network is not None:
        from ....ops.radial import compose_radial

        nodes, score, value, specs, *settings = build(
            metadata,
            projection.shape[0],
            radial_basis.shape[1],
            inputs[0].shape[1],
            (module.lmax + 1) ** 2 if wigner.ndim == 2 else wigner_inv.shape[1],
            wigner.ndim == 2,
        )
        program, inputs = compose_radial(
            repr((nodes, ((score, 0, "edge"), (value, 1, "edge")))),
            inputs,
            1,
            radial_network,
        )
        nodes, outputs = parse_metadata(program)
        specs = list(specs) + [None] * (len(inputs) - len(specs))
        for op, width, _, data in nodes:
            if op == "input":
                specs[data[0]] = (width, data[1])
        base = (nodes, outputs[0][0], outputs[1][0], tuple(specs), *settings)
        metadata = repr((metadata, base))
    result = interaction(
        metadata,
        edge_index[0],
        edge_index[1],
        inputs,
    )[0]
    return module.reshape_out.inverse(result)
