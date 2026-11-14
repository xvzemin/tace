"""Symmetric polynomial contractions and an independent monomial reference."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.metadata import parse_metadata
from .... import config, get_kernel
from ....common.ace import ace_inputs
from ....common.symmetric import symmetric_inputs, symmetric_result
from ....common.torch.linear import grouped_linear_available
from ....common.torch.symmetric import evaluate_prefix, prefix_plan
from .ictd_symmetric_contraction import project


def reference(module, features, weight, node_type):
    """Multiply each declared monomial and scatter its coefficient separately."""
    value = features.transpose(1, 2)
    output = value.new_zeros((value.shape[0], module.output_dim, module.num_channels))
    selected = weight[node_type.long()]
    for degree in range(1, module.correlation + 1):
        indices = getattr(module, f"indices_{degree}")
        if not indices.shape[0]:
            continue
        product = value[:, indices[:, 0], :]
        for k in range(1, degree):
            product = product * value[:, indices[:, k], :]
        product = product * selected[:, getattr(module, f"weights_{degree}"), :]
        product = (
            product
            * getattr(module, f"coefficients_{degree}").to(value.dtype)[None, :, None]
        )
        output = output.index_add(1, getattr(module, f"outputs_{degree}"), product)
    if not module.has_terms:
        output = output + (value.sum() + weight.sum()) * 0
    return symmetric_result(output, module.layout, module.output_dims)


def prefix_enabled(module, features, program):
    """Select the measured CPU schedules using static model limits in graphs."""
    if (
        not config.TORCH_SYMMETRIC_PREFIX
        or config.TORCH_RECOMPUTE
        or features.device.type != "cpu"
        or not 2 <= module.correlation <= 4
    ):
        return False
    terms = len(parse_metadata(module.metadata)[-1])
    if terms < 8:
        return False
    if (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or is_fake(features)
    ):
        if module.num_channels < 64:
            return False
        if terms <= config.SYMMETRIC_PREFIX_MAX_TERMS:
            return True
        budget = config.SYMMETRIC_PREFIX_MAX_NODES
        return (
            terms <= budget and len(prefix_plan(module.metadata, program)[0]) <= budget
        )
    return (
        terms <= config.SYMMETRIC_PREFIX_MAX_TERMS
        and module.correlation >= 3
        and features.shape[0] * module.num_channels * terms
        >= config.SYMMETRIC_PREFIX_MIN_WORK
    )


def uuu(module, features, weight, node_type):
    """Evaluate the scheduled polynomial without materialized tensor powers."""
    if config.TORCH_REFERENCE:
        return reference(module, features, weight, node_type)
    program, operands = symmetric_inputs(
        features, weight, module.correlation, module.output_dim
    )
    kernel = (
        evaluate_prefix
        if prefix_enabled(module, features, program)
        else get_kernel("symmetric_contract", "torch")
    )
    output = kernel(module.metadata, program, node_type, operands)[0]
    return symmetric_result(output, module.layout, module.output_dims)


def linear(module, features, weight, node_type, stage):
    """Share element matrices across nodes instead of gathering dense weights."""
    coefficient = module.coefficient_maps[stage]
    if coefficient is not None and (
        hasattr(module.linears[stage], "output_basis")
        or module.linears[stage].shared_weights
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
        return output if projection is None else project(projection, output)
    return module.linears[stage](features, weight[node_type])


def uuu_fully_connected(module, features, weights, node_type):
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
