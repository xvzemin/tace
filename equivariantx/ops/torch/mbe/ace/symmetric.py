"""Symmetric polynomial contractions and an independent monomial reference."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.metadata import parse_metadata
from .... import config, get_kernel
from ....common.symmetric import symmetric_inputs, symmetric_result
from ....common.torch.symmetric import evaluate_prefix, prefix_plan


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


def evaluate(module, features, weight, node_type):
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
