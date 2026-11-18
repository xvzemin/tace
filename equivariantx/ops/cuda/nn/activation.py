"""Compiler-visible fused Clenshaw evaluation and exact polynomial adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...precision import validate_uniform
from ...torch.nn import activation as torch_kernel


@torch.library.custom_op("eqx::block_activation", mutates_args=(), device_types="cuda")
def block_activation(
    x: torch.Tensor,
    gradient: torch.Tensor | None,
    sizes: list[int],
    kinds: list[int],
    scales: list[float],
) -> torch.Tensor:
    """Fuse normalized scalar activations or their vector-Jacobian product."""
    validate_uniform((x,) if gradient is None else (x, gradient))
    if (
        sum(sizes) != x.shape[-1]
        or len(sizes) != len(kinds)
        or len(kinds) != len(scales)
        or any(k not in range(4) for k in kinds)
        or (gradient is not None and x.shape != gradient.shape)
    ):
        raise ValueError("Scalar activation blocks and operands must match.")
    x = x.contiguous()
    result = torch.empty_like(x)
    if not result.numel():
        return result
    expressions = (
        ("1", "s+v*s*(1-s)", "s*(1-s)", "1-s*s")
        if gradient is not None
        else ("v", "v*s", "s", "s")
    )
    body = [f"int c=i%{sum(sizes)}; T v=p0[i];"]
    end = 0
    for size, kind, scale in zip(sizes, kinds, scales):
        end += size
        s = (
            "",
            "T s=T(1)/(T(1)+exp(-v));",
            "T s=T(1)/(T(1)+exp(-v));",
            "T s=tanh(v);",
        )[kind]
        factor = "p1[i]*" if gradient is not None else ""
        body.append(
            f"if(c<{end}){{{s}out[i]={factor}T({scale!r})*({expressions[kind]});continue;}}"
        )
    inputs = (x,) if gradient is None else (x, gradient.contiguous())
    return pointwise(result, inputs, "\n".join(body))


@block_activation.register_fake
def block_fake(x, gradient, sizes, kinds, scales):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def block_context(ctx, inputs, output):
    x, gradient, ctx.sizes, ctx.kinds, ctx.scales = inputs
    ctx.save_for_backward(x, gradient)


def block_backward(ctx, grad):
    x, gradient = ctx.saved_tensors
    if gradient is None:
        return (
            block_activation(x, grad, ctx.sizes, ctx.kinds, ctx.scales),
            None,
            None,
            None,
            None,
        )
    groups = tuple(zip(ctx.sizes, ctx.kinds, ctx.scales))
    dx = (
        torch_kernel.scalar_values(x, groups, grad * gradient, 2)
        if ctx.needs_input_grad[0]
        else None
    )
    dg = (
        torch_kernel.scalar_values(x, groups, grad, 1)
        if ctx.needs_input_grad[1]
        else None
    )
    return dx, dg, None, None, None


block_activation.register_autograd(block_backward, setup_context=block_context)


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
