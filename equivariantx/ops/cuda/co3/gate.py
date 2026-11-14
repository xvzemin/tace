"""Fused gate expressions in the module's representation layout."""

from ...common.gate import gate


def evaluate(module, features):
    """Apply the native expression or its compiler-visible Torch lowering."""
    return gate(features, module.metadata, module.irreps_out.dim)
