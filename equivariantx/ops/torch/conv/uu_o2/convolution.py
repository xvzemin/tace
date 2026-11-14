"""Channelwise local weight maps and differentiable Wigner evaluation."""

import torch

from .... import config
from ...o2 import wigner as torch_wigner


def transform_weights(module, weights):
    """Transform channelwise weights into transverse coupling paths."""
    values = weights.reshape(
        weights.shape[0], module.weight_numel // module.num_channel, module.num_channel
    )
    result = []
    for index, (shape, _) in enumerate(module.weight_transforms):
        count, outputs, inputs = shape
        selected = values.index_select(1, getattr(module, f"weight_index_{index}"))
        selected = selected.reshape(weights.shape[0], count, inputs, module.num_channel)
        result.append(
            torch.einsum(
                "goi,bgic->bgoc", getattr(module, f"weight_transform_{index}"), selected
            ).reshape(weights.shape[0], count * outputs * module.num_channel)
        )
    return torch.cat(result, dim=-1) if result else weights[..., :0]


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
    """Evaluate the parent contraction with differentiable alignment matrices."""
    from equivariantx.conv.o2_o3.convolution import O2O3TensorProductConv

    if vectors is not None:
        wigner = (
            torch_wigner.packed(module.frame, vectors, "recursive")
            if module.backend == "torch" and config.TORCH_REFERENCE
            else module.frame.forward_packed(vectors, method="recursive")
        )
    result = O2O3TensorProductConv.forward(
        module,
        features,
        radial,
        projection,
        wigner,
        cutoff,
        edge_index,
        num_nodes,
        radial_network=radial_network,
    )
    return result if vectors is None else result + vectors.sum() * 0
