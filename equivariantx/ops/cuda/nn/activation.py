"""Compiler-visible fused Clenshaw evaluation and exact polynomial adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...precision import validate_uniform
from ...torch.nn import activation as torch_kernel


@torch.library.custom_op("eqx::chebyshev", mutates_args=(), device_types="cuda")
def chebyshev(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate a Chebyshev series using the Clenshaw recurrence."""
    validate_uniform((features, coefficients))
    if coefficients.ndim != 1 or not coefficients.numel():
        raise ValueError("Chebyshev coefficients must be a nonempty vector.")
    features, coefficients = features.contiguous(), coefficients.contiguous()
    result = torch.empty_like(features)
    body = f"""
        T x=p0[i],previous=0,current=0;
        for(int k={coefficients.numel() - 1};k>0;--k) {{
            T value=p1[k]+T(2)*x*current-previous;
            previous=current; current=value;
        }}
        out[i]=p1[0]+x*current-previous;
    """
    return pointwise(result, (features, coefficients), body)


@chebyshev.register_fake
def fake(features, coefficients):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


chebyshev.register_autograd(
    torch_kernel.Chebyshev.backward, setup_context=setup_context
)


def evaluate(module, features):
    """Fuse eager evaluation and expose the complete polynomial for compilation."""
    if (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or forward_ad_active()
    ):
        return torch_kernel.evaluate(module, features)
    return chebyshev(features / module.bound, module.coefficients)
