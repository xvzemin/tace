"""Indexed channel maps with bounded packing and an ordinary Torch oracle."""

from ...common.torch.linear import (
    element_linear,
    element_linear_reference,
    indexed_parameters,
)


def reference(module, features, weight, node_type, target_type=None):
    """Gather one matrix per sample and use ordinary batched matrix products."""
    weight, node_type = indexed_parameters(
        weight, node_type, target_type, module.num_experts, module.weight_numel
    )
    return element_linear_reference(
        features, weight, node_type, module.paths, module.output_dim, module.layout
    )


def evaluate(module, features, weight, node_type, target_type=None):
    """Share indexed matrices across bounded groups of samples."""
    weight, node_type = indexed_parameters(
        weight, node_type, target_type, module.num_experts, module.weight_numel
    )
    return element_linear(
        features,
        weight,
        node_type,
        module.paths,
        module.output_dim,
        module.layout,
        module.expand_weights,
    )
