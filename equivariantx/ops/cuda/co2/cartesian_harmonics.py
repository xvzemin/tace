"""Native planar Cartesian harmonics and exact compact-space adjoints."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...common.cuda.harmonics import complex_multiply
from ...precision import validate_precision
from ...torch.co2 import cartesian_harmonics as torch_kernel
from ...torch.o2.circular_harmonics import Circular


@triton.jit
def cartesian_kernel(
    source,
    output,
    size,
    WIDTH: tl.constexpr,
    ORDERS: tl.constexpr,
    SCALES: tl.constexpr,
    OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // WIDTH, index % WIDTH
    valid = index < size
    x = tl.load(source + row * 2, valid, 0)
    y = tl.load(source + row * 2 + 1, valid, 0)
    for i in tl.static_range(len(ORDERS)):
        component = column - OFFSETS[i]
        selected = valid & (component >= 0) & (component < 2 ** ORDERS[i])
        real = tl.full((BLOCK,), 1, x.dtype)
        imaginary = tl.full((BLOCK,), 0, x.dtype)
        phase = tl.full((BLOCK,), 0, tl.int32)
        for j in tl.static_range(ORDERS[i]):
            real, imaginary = complex_multiply(real, imaginary, x, y)
            phase += (component >> j) & 1
        value = tl.where(phase % 2 == 0, real, imaginary)
        value = tl.where(phase % 4 < 2, value, -value) * SCALES[i]
        tl.store(output + index, value, selected)


@triton_op("eqx::planar_cartesian_harmonics", mutates_args=())
def cartesian(
    vectors: torch.Tensor, orders: list[int], normalization: str
) -> torch.Tensor:
    """Fuse circular powers and signed Cartesian expansion in one launch."""
    validate_precision((vectors,))
    if not vectors.is_cuda or vectors.ndim < 1 or vectors.shape[-1] != 2:
        raise ValueError("Native planar harmonics require CUDA vectors of size two.")
    if any(order < 0 for order in orders) or normalization not in (
        "norm",
        "component",
        "integral",
    ):
        raise ValueError("Invalid harmonic order or normalization.")
    width = sum(2**order for order in orders)
    vectors = vectors.contiguous()
    result = vectors.new_empty((*vectors.shape[:-1], width))
    if result.numel():
        wrap_triton(cartesian_kernel)[(triton.cdiv(result.numel(), 256),)](
            vectors,
            result,
            result.numel(),
            width,
            tuple(orders),
            tuple(torch_kernel.scale(order, normalization) for order in orders),
            tuple(sum(2**degree for degree in orders[:i]) for i in range(len(orders))),
            256,
            enable_fp_fusion=False,
        )
    return result


def setup_context(ctx, inputs, output):
    vectors, ctx.orders, ctx.normalization = inputs
    ctx.mmax = max(ctx.orders, default=0)
    ctx.save_for_backward(vectors)


def backward(ctx, gradient):
    """Reduce Cartesian cotangents before the recursive circular adjoint."""
    (vectors,) = ctx.saved_tensors
    indices, scales = torch_kernel.layout(tuple(ctx.orders), ctx.normalization)
    index = torch.tensor(indices, device=vectors.device, dtype=torch.int64)
    scale = vectors.new_tensor(scales)
    compact = vectors.new_zeros((*vectors.shape[:-1], 2 * ctx.mmax + 1))
    compact = compact.index_add(-1, index, gradient * scale)
    return Circular.backward(ctx, compact)[0], None, None


cartesian.register_autograd(backward, setup_context=setup_context)


def evaluate(module, vectors):
    """Fuse eager powers/expansion and expose exact Torch algebra to AOTI."""
    if (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or forward_ad_active()
    ):
        return torch_kernel.evaluate(module, vectors)
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return cartesian(vectors, list(module.orders), module.normalization)
