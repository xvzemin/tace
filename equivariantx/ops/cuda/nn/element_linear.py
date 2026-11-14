"""Indexed channel maps using the native bilinear contraction and its adjoints."""

from ...common.linear import contract
from ...common.torch.linear import indexed_parameters


def evaluate(module, features, weight, node_type, target_type=None):
    """Evaluate independent single-element or ordered-pair channel maps."""
    weight, node_type = indexed_parameters(
        weight, node_type, target_type, module.num_experts, module.weight_numel
    )
    return contract(
        module.metadata,
        2,
        node_type,
        features,
        weight,
        features.new_empty((0, module.output_dim)),
    )
