"""Native low-order Cartesian harmonics and exact higher-order derivatives."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...precision import validate_precision
from ...torch.co3 import cartesian_harmonics as torch_kernel


@torch.library.custom_op(
    "eqx::cartesian_harmonics_low", mutates_args=(), device_types="cuda"
)
def low_order(
    vectors: torch.Tensor, orders: list[int], normalization: str
) -> torch.Tensor:
    """Evaluate Cartesian harmonics of degree zero, one and two."""
    validate_precision((vectors,))
    if (
        vectors.ndim < 1
        or vectors.shape[-1] != 3
        or any(l not in (0, 1, 2) for l in orders)
    ):
        raise ValueError(
            "Low-order harmonics require spatial vectors and degrees 0..2."
        )
    if normalization not in ("norm", "component", "integral"):
        raise ValueError("Unknown harmonic normalization.")
    vectors = vectors.contiguous()
    width = sum(3**l for l in orders)
    result = vectors.new_empty((*vectors.shape[:-1], width))
    if not result.numel():
        return result
    cases, offset = [], 0
    for degree in orders:
        scale = torch_kernel.scale(degree, normalization)
        expression = (
            "T(1)"
            if degree == 0
            else f"p0[row*3+c-{offset}]"
            if degree == 1
            else f"(p0[row*3+(c-{offset})/3]*p0[row*3+(c-{offset})%3]-(((c-{offset})/3==(c-{offset})%3)?trace:T(0)))*T(1.224744871391589)"
        )
        cases.append(
            f"if(c>={offset} && c<{offset + 3**degree}) out[i]=({expression})*T({scale:.17g});"
        )
        offset += 3**degree
    body = f"I row=i/{width}; int c=i%{width}; T x=p0[3*row],y=p0[3*row+1],z=p0[3*row+2]; T trace=(x*x+y*y+z*z)/T(3); {''.join(cases)}"
    return pointwise(result, (vectors,), body)


@low_order.register_fake
def fake(vectors, orders, normalization):
    return vectors.new_empty((*vectors.shape[:-1], sum(3**l for l in orders)))


def setup_context(ctx, inputs, output):
    vectors, ctx.orders, ctx.normalization = inputs
    ctx.save_for_backward(vectors)


def backward(ctx, gradient):
    """Apply the exact polynomial Jacobian transpose using differentiable Torch."""
    (vectors,) = ctx.saved_tensors
    result = vectors * 0
    blocks = gradient.split([3**degree for degree in ctx.orders], -1)
    for degree, value in zip(ctx.orders, blocks):
        scale = torch_kernel.scale(degree, ctx.normalization)
        if degree == 1:
            result = result + value * scale
        elif degree == 2:
            value = value.unflatten(-1, (3, 3))
            symmetric = value + value.transpose(-1, -2)
            trace = value.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
            result = result + math.sqrt(1.5) * scale * (
                (symmetric * vectors.unsqueeze(-2)).sum(-1) - (2 / 3) * trace * vectors
            )
    return result, None, None


low_order.register_autograd(backward, setup_context=setup_context)


def evaluate(module, vectors):
    """Use a native eager kernel and expose complete algebra when compiling."""
    if (
        module.lmax > 2
        or torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or forward_ad_active()
    ):
        return torch_kernel.evaluate(module, vectors)
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return low_order(vectors, list(torch_kernel.degrees(module)), module.normalization)
