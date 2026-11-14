"""Recursive products and indexed coefficient maps for each body order."""

from .... import config, get_kernel
from ....common.ace import ace_inputs
from ....common.torch.linear import grouped_linear_available


def linear(module, features, weight, node_type, stage):
    """Share element matrices across nodes instead of gathering dense weights."""
    coefficient = module.coefficient_maps[stage]
    if coefficient is not None and (
        module.linears[stage].shared_weights
        or (
            not config.TORCH_REFERENCE
            and (
                features.is_cuda
                or grouped_linear_available(features, weight, coefficient.paths)
            )
        )
    ):
        output = coefficient(features, weight, node_type)
        projection = getattr(module.linears[stage], "projection", None)
        return output if projection is None else projection(output)
    return module.linears[stage](features, weight[node_type])


def evaluate(module, features, weights, node_type):
    """Accumulate each body order while reusing the preceding product."""
    correlation = features
    output = linear(module, features, weights[0], node_type, 0)
    for stage, weight in enumerate(weights[1:]):
        if getattr(module.tensor_products[stage], "backend", "torch") != "torch":
            final = stage == len(module.product_dims) - 1
            program, operands = ace_inputs(
                correlation,
                features,
                weight if final else None,
                module.contraction_dim if final else module.product_dims[stage],
            )
            value = get_kernel("ace_contract", "torch")(
                module.metadata[stage], program, node_type, operands
            )[0]
            if final:
                output = output + module.project_output(value)
                continue
            correlation = module.project_product(value, stage)
        else:
            correlation = module.tensor_products[stage](correlation, features)
        output = output + linear(module, correlation, weight, node_type, stage + 1)
    return output
