"""Dense and recursive contraction of circular O(2) coordinates."""

import torch

from ... import config
from ...common.torch.o2_contraction import Contraction, Coupling, Readout

__all__ = ["Contraction", "Coupling", "Readout", "evaluate", "dense"]


def dense(module, inputs, order_weights):
    letters = "abdefghijklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if module.correlation > len(letters):
        raise ValueError("correlation exceeds the number of einsum indices.")
    output = None
    for order_index in range(module.correlation):
        order = order_index + 1
        indices = letters[:order]
        coefficient = getattr(module, f"generalized_cg_{order}").to(inputs[0])
        # Weight the generalized tensor, then contract one input at a time.
        contribution = torch.einsum(
            f"o{indices}p,...pc->...o{indices}c",
            coefficient,
            order_weights[order_index],
        )
        for i in reversed(range(order)):
            contribution = torch.einsum(
                f"...o{indices[: i + 1]}c,...{indices[i]}c->...o{indices[:i]}c",
                contribution,
                inputs[i],
            )
        output = contribution if output is None else output + contribution
    return module._flatten_output(output)


def prepare(module, inputs, weight, leading_shape):
    """Keep flattened irrep layouts and broadcasts outside contraction kernels."""
    inputs = tuple(
        features.expand(*leading_shape, module.input_dim).reshape(
            *leading_shape, module._base_input_dim, module.num_channels
        )
        for features in (inputs[:1] if module.shared_inputs else inputs)
    )
    if module.shared_inputs:
        inputs = inputs * module.correlation
    order_weights = tuple(
        weight[..., section].reshape(*weight.shape[:-1], count, module.num_channels)
        for section, count in zip(module._order_weight_slices, module.order_num_paths)
    )
    return inputs, order_weights


def evaluate(module, inputs, weight, leading_shape, *, backend="torch", optimized=None):
    inputs, order_weights = prepare(module, inputs, weight, leading_shape)
    if module.algorithm == "dense":
        return dense(module, inputs, order_weights)
    if optimized is None:
        optimized = not config.TORCH_REFERENCE and getattr(
            config, "TORCH_O2_CONTRACTION", False
        )
    all_weights = (
        weight.unflatten(-1, (module.num_paths, module.num_channels))
        if module.contraction.joint is not None
        else None
    )
    output = module.contraction(
        inputs, order_weights, backend, optimized, all_weights=all_weights
    )
    return module._flatten_output(output)
