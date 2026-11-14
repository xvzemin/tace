"""Native fused gated products and recursive transposed programs."""

from ....common.bilinear import bilinear_inputs, contract


def evaluate(module, features, base, gates, weight, node_type, shared_weight=None):
    """Fuse expert and shared coefficient contributions into the output."""
    program, operands = bilinear_inputs(
        features, base, gates, weight, module.contraction_dim, shared_weight
    )
    output = contract(module.kernel_metadata, program, node_type, operands)[0]
    return module.project_output(output)
