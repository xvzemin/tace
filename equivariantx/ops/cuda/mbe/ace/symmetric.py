"""Native symmetric contractions with recursively differentiated programs."""

from ....common.symmetric import symmetric_contract, symmetric_inputs, symmetric_result


def evaluate(module, features, weight, node_type):
    """Contract directly into the declared angular/channel output layout."""
    program, operands = symmetric_inputs(
        features, weight, module.correlation, module.output_dim
    )
    output = symmetric_contract(module.metadata, program, node_type, operands)[0]
    return symmetric_result(output, module.layout, module.output_dims)
