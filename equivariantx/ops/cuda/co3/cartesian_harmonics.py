"""Native low-order Cartesian harmonics and exact higher-order derivatives."""

import math

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...precision import validate_precision
from ...torch.co3 import cartesian_harmonics as torch_kernel


@triton.jit
def harmonic_component(x, y, z, component, DEGREE: tl.constexpr, SCALE: tl.constexpr):
    if DEGREE == 0:
        return tl.full(x.shape, SCALE, x.dtype)
    if DEGREE == 1:
        return tl.where(component == 0, x, tl.where(component == 1, y, z)) * SCALE
    i, j = component // 3, component % 3
    a = tl.where(i == 0, x, tl.where(i == 1, y, z))
    b = tl.where(j == 0, x, tl.where(j == 1, y, z))
    trace = (x * x + y * y + z * z) / 3.0
    return (a * b - tl.where(i == j, trace, 0.0)) * (1.224744871391589 * SCALE)


@triton.jit
def harmonic_kernel(
    source,
    output,
    size,
    WIDTH: tl.constexpr,
    ORDERS: tl.constexpr,
    SCALES: tl.constexpr,
    OFFSETS: tl.constexpr,
    WIDTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // WIDTH, index % WIDTH
    valid = index < size
    x = tl.load(source + 3 * row, valid, 0)
    y = tl.load(source + 3 * row + 1, valid, 0)
    z = tl.load(source + 3 * row + 2, valid, 0)
    for k in tl.static_range(len(ORDERS)):
        component = column - OFFSETS[k]
        selected = valid & (component >= 0) & (component < WIDTHS[k])
        value = harmonic_component(x, y, z, component, ORDERS[k], SCALES[k])
        tl.store(output + index, value, selected)


@triton_op("eqx::cartesian_harmonics_low", mutates_args=())
def low_order(
    vectors: torch.Tensor, orders: list[int], normalization: str
) -> torch.Tensor:
    """Evaluate contiguous vectors in one CUDA launch with FP32/FP64 storage."""
    validate_precision((vectors,))
    if not vectors.is_cuda or vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError(
            "Native harmonics require CUDA vectors with trailing size three."
        )
    if any(degree not in (0, 1, 2) for degree in orders):
        raise ValueError("Native low-order harmonics require degrees zero through two.")
    if normalization not in ("norm", "component", "integral"):
        raise ValueError("Unknown harmonic normalization.")
    vectors = vectors.contiguous()
    width = sum(3**degree for degree in orders)
    result = vectors.new_empty((*vectors.shape[:-1], width))
    if result.numel():
        scales = tuple(torch_kernel.scale(degree, normalization) for degree in orders)
        wrap_triton(harmonic_kernel)[(triton.cdiv(result.numel(), 256),)](
            vectors,
            result,
            result.numel(),
            width,
            tuple(orders),
            scales,
            tuple(sum(3**degree for degree in orders[:i]) for i in range(len(orders))),
            tuple(3**degree for degree in orders),
            256,
        )
    return result


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
