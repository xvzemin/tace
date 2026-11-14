"""Native symmetric contractions with recursively differentiated programs."""

from ....common.ace import ace_inputs, contract
from ....common.symmetric import symmetric_contract, symmetric_inputs, symmetric_result


def uuu(module, features, weight, node_type):
    """Contract directly into the declared angular/channel output layout."""
    program, operands = symmetric_inputs(
        features, weight, module.correlation, module.output_dim
    )
    output = symmetric_contract(module.metadata, program, node_type, operands)[0]
    return symmetric_result(output, module.layout, module.output_dims)


def linear(module, features, weight, node_type, stage):
    """Apply the native element-dependent coefficient contraction."""
    operands = [
        features,
        features.new_ones((features.shape[0], 1)),
        weight,
        features.new_empty((0, module.contraction_dim)),
    ]
    return contract(
        module.linear_metadata[stage],
        repr((((0, 1, 2, 3), 3, 0),)),
        node_type,
        operands,
    )[0]


def uuu_fully_connected(module, features, weights, node_type):
    """Accumulate native contractions without forming the final product tensor."""
    correlation = features
    output = linear(module, features, weights[0], node_type, 0)
    for stage, weight in enumerate(weights[1:]):
        final = stage == len(module.product_dims) - 1
        program, operands = ace_inputs(
            correlation,
            features,
            weight if final else None,
            module.contraction_dim if final else module.product_dims[stage],
        )
        value = contract(module.metadata[stage], program, node_type, operands)[0]
        if final:
            output = output + module.project_output(value)
        else:
            correlation = module.project_product(value, stage)
            output = output + linear(module, correlation, weight, node_type, stage + 1)
    return output
