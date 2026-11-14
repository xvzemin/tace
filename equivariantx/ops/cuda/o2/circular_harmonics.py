"""Native circular powers with an exact differentiable analytic adjoint."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...precision import validate_precision
from ...torch.o2 import circular_harmonics as torch_kernel


@torch.library.custom_op(
    "eqx::circular_harmonics", mutates_args=(), device_types="cuda"
)
def circular(vectors: torch.Tensor, mmax: int) -> torch.Tensor:
    """Evaluate real and imaginary powers with CUDA arithmetic."""
    validate_precision((vectors,))
    if vectors.ndim < 1 or vectors.shape[-1] != 2 or mmax < 0:
        raise ValueError(
            "Circular harmonics require planar vectors and a non-negative order."
        )
    vectors = vectors.contiguous()
    result = vectors.new_empty((*vectors.shape[:-1], 2 * mmax + 1))
    body = f"""
        I row=i/{2 * mmax + 1}; int c=i%{2 * mmax + 1};
        T x=p0[row*2],y=p0[row*2+1],re=1,im=0;
        for(int m=0;m<(c+1)/2;++m) {{
            T next_re=re*x-im*y;
            im=re*y+im*x; re=next_re;
        }}
        out[i]=(c && !(c%2)) ? im : re;
    """
    return pointwise(result, (vectors,), body)


@circular.register_fake
def fake(vectors, mmax):
    return vectors.new_empty((*vectors.shape[:-1], 2 * mmax + 1))


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
