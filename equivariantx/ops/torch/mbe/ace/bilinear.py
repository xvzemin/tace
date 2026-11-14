"""Gated products with element-dependent and shared channel coefficients."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config, get_kernel
from ....common.bilinear import bilinear_inputs
from ....common.torch.ace import bilinear_reference
from ....common.torch.linear import SharedCoefficients, element_linear_blocks


def reference(module, features, base, gates, weight, node_type, shared_weight=None):
    """Evaluate independent angular paths with ordinary Torch contractions."""
    program, operands = bilinear_inputs(
        features, base, gates, weight, module.contraction_dim, shared_weight
    )
    output = bilinear_reference(module.kernel_metadata, program, node_type, operands)[0]
    return module.project_output(output)


def shared_coefficients_available(module, features, weight):
    """Merge matching maps only for measured, statically sized CPU programs."""
    if (
        not config.TORCH_SHARED_COEFFICIENTS
        or features.device.type != "cpu"
        or module.shared is None
        or not (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return False
    nodes, elements = features.shape[0], weight.shape[0]
    if isinstance(nodes, torch.SymInt) or isinstance(elements, torch.SymInt):
        return False
    element, shared = module.coefficients, module.shared.coefficients
    if (
        nodes < 256
        or not 1 <= elements <= 16
        or element.num_experts != 1
        or element.layout != shared.layout
        or element.output_dim != shared.output_dim
        or element.paths != shared.paths
        or not element.paths
    ):
        return False
    if not all(64 <= ci == co <= 1024 for _, _, _, ci, co, _, _ in element.paths):
        return False
    workspace = elements * sum(ci * co for _, _, _, ci, co, _, _ in element.paths)
    return workspace * weight.element_size() <= config.SHARED_COEFFICIENTS_MAX_BYTES


def evaluate(module, features, base, gates, weight, node_type, shared_weight=None):
    """Reuse products and indexed coefficient maps when their layout permits."""
    if config.TORCH_REFERENCE:
        return reference(
            module, features, base, gates, weight, node_type, shared_weight
        )
    if (
        getattr(module.tensor_products[0], "backend", "torch") != "torch"
        or module.tensor_products[0].shared_weights
    ):
        program, operands = bilinear_inputs(
            features, base, gates, weight, module.contraction_dim, shared_weight
        )
        output = get_kernel("bilinear_ace", "torch")(
            module.kernel_metadata, program, node_type, operands
        )[0]
        return module.project_output(output)
    product = module.tensor_products[0](features, base, gates)
    if shared_coefficients_available(module, product, weight):
        merged = SharedCoefficients.apply(
            weight.reshape(weight.shape[0], -1), shared_weight.reshape(-1)
        )
        return module.coefficients(product, merged, node_type)
    output = module.coefficients(product, weight, node_type)
    if module.shared is not None:
        shared = module.shared.coefficients
        if features.device.type == "cpu":
            output = output + element_linear_blocks(
                product.unsqueeze(0),
                shared_weight.reshape(1, 1, -1),
                shared.paths,
                shared.output_dim,
                shared.layout,
            ).squeeze(0)
        else:
            output = output + shared(
                product, shared_weight.reshape(1, 1, -1), torch.zeros_like(node_type)
            )
    return output
