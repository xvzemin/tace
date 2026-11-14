"""Compiler-visible fused Clenshaw evaluation and exact polynomial adjoints."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ...torch.nn import activation as torch_kernel


@triton.jit
def clenshaw_kernel(
    source, coefficients, output, size, DEGREE: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < size
    x = tl.load(source + index, valid, 0)
    previous = tl.full((BLOCK,), 0, x.dtype)
    current = tl.full((BLOCK,), 0, x.dtype)
    for degree in tl.static_range(DEGREE, 0, -1):
        next_value = tl.load(coefficients + degree) + 2 * x * current - previous
        previous = current
        current = next_value
    value = tl.load(coefficients) + x * current - previous
    tl.store(output + index, value, valid)


@triton_op("eqx::chebyshev", mutates_args=())
def chebyshev(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate a fixed or trainable Chebyshev series in FP32 or FP64."""
    validate_uniform((features, coefficients))
    if not features.is_cuda or coefficients.ndim != 1 or not coefficients.numel():
        raise ValueError(
            "Native Chebyshev evaluation requires CUDA and a coefficient vector."
        )
    features, coefficients = features.contiguous(), coefficients.contiguous()
    result = torch.empty_like(features)
    if result.numel():
        wrap_triton(clenshaw_kernel)[(triton.cdiv(result.numel(), 256),)](
            features,
            coefficients,
            result,
            result.numel(),
            coefficients.numel() - 1,
            256,
            enable_fp_fusion=False,
        )
    return result


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
