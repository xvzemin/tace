"""Native circular powers with an exact differentiable analytic adjoint."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...common.cuda.harmonics import complex_multiply
from ...precision import validate_precision
from ...torch.o2 import circular_harmonics as torch_kernel


@triton.jit
def circular_kernel(source, output, rows, ORDER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    x = tl.load(source + row * 2, valid, 0)
    y = tl.load(source + row * 2 + 1, valid, 0)
    real = tl.full((BLOCK,), 1, x.dtype)
    imaginary = tl.full((BLOCK,), 0, x.dtype)
    tl.store(output + row * (2 * ORDER + 1), real, valid)
    for degree in tl.static_range(1, ORDER + 1):
        real, imaginary = complex_multiply(real, imaginary, x, y)
        tl.store(output + row * (2 * ORDER + 1) + 2 * degree - 1, real, valid)
        tl.store(output + row * (2 * ORDER + 1) + 2 * degree, imaginary, valid)


@triton_op("eqx::circular_harmonics", mutates_args=())
def circular(vectors: torch.Tensor, mmax: int) -> torch.Tensor:
    """Evaluate real/imaginary powers in one compiler-visible CUDA launch."""
    validate_precision((vectors,))
    if not vectors.is_cuda or vectors.ndim < 1 or vectors.shape[-1] != 2:
        raise ValueError("Native circular harmonics require CUDA vectors of size two.")
    if mmax < 0:
        raise ValueError("The maximum order must be non-negative.")
    vectors = vectors.contiguous()
    result = vectors.new_empty((*vectors.shape[:-1], 2 * mmax + 1))
    rows = vectors.numel() // 2
    if rows:
        wrap_triton(circular_kernel)[(triton.cdiv(rows, 256),)](
            vectors, result, rows, mmax, 256, enable_fp_fusion=False
        )
    return result


def setup_context(ctx, inputs, output):
    vectors, ctx.mmax = inputs
    ctx.save_for_backward(vectors)


circular.register_autograd(torch_kernel.Circular.backward, setup_context=setup_context)


def evaluate(vectors, mmax, normalize):
    """Fuse eager powers and expose ordinary algebra for joint compilation."""
    if (
        not mmax
        or torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or forward_ad_active()
    ):
        return torch_kernel.evaluate(vectors, mmax, normalize)
    if normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return circular(vectors, mmax)
