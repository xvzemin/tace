"""Native planar Cartesian harmonics and exact compact-space adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...precision import validate_precision
from ...torch.co2 import cartesian_harmonics as torch_kernel
from ...torch.o2.circular_harmonics import Circular


@torch.library.custom_op(
    "eqx::planar_cartesian_harmonics", mutates_args=(), device_types="cuda"
)
def cartesian(
    vectors: torch.Tensor, orders: list[int], normalization: str
) -> torch.Tensor:
    """Fuse circular powers and signed Cartesian expansion."""
    validate_precision((vectors,))
    if vectors.ndim < 1 or vectors.shape[-1] != 2 or any(m < 0 for m in orders):
        raise ValueError(
            "Planar harmonics require planar vectors and non-negative orders."
        )
    if normalization not in ("norm", "component", "integral"):
        raise ValueError("Unknown harmonic normalization.")
    width = sum(2**m for m in orders)
    vectors = vectors.contiguous()
    result = vectors.new_empty((*vectors.shape[:-1], width))
    if not result.numel():
        return result
    cases, offset = [], 0
    for order in orders:
        scale = torch_kernel.scale(order, normalization)
        cases.append(f"""
            if(c>={offset} && c<{offset + 2**order}) {{
                T re=1,im=0;
                for(int m=0;m<{order};++m) {{
                    T next_re=re*x-im*y; im=re*y+im*x; re=next_re;
                }}
                int phase=__popc(unsigned(c-{offset}))%4;
                out[i]=(phase%2?im:re)*T(phase<2?1:-1)*T({scale:.17g});
            }}
        """)
        offset += 2**order
    return pointwise(
        result,
        (vectors,),
        f"I row=i/{width}; int c=i%{width}; T x=p0[row*2],y=p0[row*2+1]; {''.join(cases)}",
    )


@cartesian.register_fake
def fake(vectors, orders, normalization):
    return vectors.new_empty((*vectors.shape[:-1], sum(2**m for m in orders)))


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
