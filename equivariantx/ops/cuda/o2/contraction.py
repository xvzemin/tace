"""Fused eager polynomials and compiler-visible shared-prefix contractions."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...torch.o2.contraction import evaluate as staged


def direct(module, inputs, weight, leading_shape):
    """Retain the existing fully fused polynomial program as a measured candidate."""
    from ...common.channel_product import polynomial_contraction

    output = polynomial_contraction(
        module.kernel_metadata, inputs, weight, leading_shape, module.irreps_out.dim
    )
    return module._flatten_output(
        output.reshape(*leading_shape, module._base_irreps_out.dim, module.num_channels)
    )


def evaluate(module, inputs, weight, leading_shape):
    compiling = torch.compiler.is_compiling() or get_proxy_mode() is not None
    if (
        not config.O2_CONTRACTION_STAGED
        or not hasattr(module, "contraction")
        or (not compiling and not config.O2_CONTRACTION_EAGER_STAGED)
    ):
        return direct(module, inputs, weight, leading_shape)
    inline = config.O2_CONTRACTION_INLINE and compiling
    return staged(
        module,
        inputs,
        weight,
        leading_shape,
        backend="torch" if inline else "cuda",
        optimized=not inline,
    )
