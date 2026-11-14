"""Native or compiler-lowered spherical gate expressions."""

from ...common.gate import gate


def evaluate(module, features):
    """Evaluate the gate in spherical multiplicity-first storage."""
    return gate(features, module.metadata, module.output_dim)
