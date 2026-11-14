"""Torch angular products, grouped channel GEMMs, and reference paths."""

from ... import config
from ...common.torch import channel_product


def evaluate(module, x, y, weight):
    """Select the ordinary reference or optimized channel schedule."""
    if not config.TORCH_REFERENCE and channel_product.use_slices(module, x, y, weight):
        return module.projection(channel_product.tensor_product(module, x, y, weight))
    contract = module.reference if config.TORCH_REFERENCE else module.contraction
    output = contract(x, y, weight, module.path_scales.to(x))
    return module.projection(output)
