"""Reciprocal radial projections for the frozen five-layer 7M force graph."""

from contextvars import ContextVar
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.o3.convolution import contract_direct
from equivariantx.ops.common.cuda.pointwise import launch
from equivariantx.ops.common.cuda.runtime import runtime

from ..tace_oam_l.tangent_kernels import adjoint_plan
from .radial_batch import batch_radial

reuse_pairs = ContextVar("eqx_7m_reciprocal_projections", default=False)
TILE = 32768


@lru_cache(maxsize=4)
def mapping_kernel(width, key_type):
    return f"""
using I=long long;
extern "C" __global__ void run(const {key_type}* key,const I* order,const I* src,
    const I* dst,const I* types,const float* f,const float* df,I* parent,I rows) {{
    I warp=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    int lane=threadIdx.x%32;
    for(I p=warp;p<rows;p+=I(gridDim.x)*blockDim.x/32) {{
        I edge=order[p],other=edge;
        if(p>0 && key[p-1]==key[p] && (p<2 || key[p-2]!=key[p])
                && (p+1>=rows || key[p+1]!=key[p])) {{
            I prev=order[p-1];
            if(src[edge]!=dst[edge] && src[prev]==dst[edge] && dst[prev]==src[edge]
                    && types[src[edge]]==types[dst[edge]]) {{
                bool equal=true;
                #pragma unroll
                for(int c=lane;c<{width};c+=32)
                    equal=equal && f[edge*{width}+c]==f[prev*{width}+c]
                                && df[edge*{width}+c]==df[prev*{width}+c];
                if(__all_sync(0xffffffff,equal)) other=prev;
            }}
        }}
        if(!lane) parent[edge]=other;
    }}
}}"""


def pair_rows(src, dst, types, f, df):
    """Return representatives and inverse rows for equal reciprocal radial states."""
    if f.dtype != torch.float32 or df.dtype != f.dtype or f.shape != df.shape:
        raise ValueError("Expected matching FP32 radial features and tangents.")
    if any(v.dtype != torch.int64 for v in (src, dst, types)):
        raise ValueError("Expected int64 edge indices and atom types.")
    src, dst, types, f, df = (v.contiguous() for v in (src, dst, types, f, df))
    n, e = len(types), len(src)
    if e == 0:
        return None
    key = torch.minimum(src, dst) * n + torch.maximum(src, dst)
    compact = n * n < 1 << 31
    keys, order = (key.int() if compact else key).sort()
    parent = torch.empty_like(src)
    launch(
        mapping_kernel(f.size(1), "int" if compact else "long long"),
        (keys, order, src, dst, types, f, df, parent),
        (e,),
        e * 32,
    )
    keep = torch.arange(e, device=src.device) == parent
    reps = keep.nonzero(as_tuple=True)[0]
    if 8 * len(reps) > 7 * e:
        return None
    return reps, (keep.cumsum(0) - 1)[parent]


@torch.library.custom_op(
    "eqx_tace_7m::reciprocal_heads", mutates_args=(), device_types="cuda"
)
def reciprocal_heads(
    f: torch.Tensor,
    df: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    types: torch.Tensor,
    p: list[torch.Tensor],
    eps1: float,
    eps2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = pair_rows(src, dst, types, f, df) if reuse_pairs.get() else None
    if rows is None:
        return (
            *batch_radial(f, df, types[src], types[dst], p, eps1, eps2),
            src.new_empty(0),
        )
    reps, inverse = rows
    h, d, dh, dd = batch_radial(
        f[reps], df[reps], types[src[reps]], types[dst[reps]], p, eps1, eps2
    )
    return h, d.index_select(1, inverse), dh, dd.index_select(1, inverse), inverse


@reciprocal_heads.register_fake
def fake_heads(f, df, src, dst, types, p, eps1, eps2):
    ctx = torch.library.get_ctx()
    n, rows, pairs = f.shape[0], ctx.new_dynamic_size(), ctx.new_dynamic_size()
    return (
        f.new_empty((5, rows, 64)),
        f.new_empty((5, n, 1)),
        f.new_empty((5, rows, 64)),
        f.new_empty((5, n, 1)),
        src.new_empty((pairs,)),
    )


@torch.library.custom_op(
    "eqx_tace_7m::reciprocal_projection", mutates_args=(), device_types="cuda"
)
def reciprocal_projection(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    shape: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    inverse: torch.Tensor,
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .tangent import projected, projection_capacity

    if not len(inverse):
        return projected(x, y, r, h, dh, src, dst, shape, w, b, meta)
    out = x.new_zeros(shape.shape)
    coeff = h.new_empty((projection_capacity(len(h)), w.size(1)))[: len(h)]
    torch.addmm(b, h, w, out=coeff)
    for lo in range(0, len(src), TILE):
        sl = slice(lo, lo + TILE)
        contract_direct(
            meta,
            src[sl],
            dst[sl],
            [((4,), (x, coeff, coeff[:0], y[sl], out), (out,), False)],
            pair_index=inverse[sl],
        )
    return out, coeff


@reciprocal_projection.register_fake
def fake_projection(x, y, r, h, dh, src, dst, shape, w, b, inverse, meta):
    return x.new_empty(shape.shape), h.new_empty(
        (torch.library.get_ctx().new_dynamic_size(), w.size(1))
    )


@torch.library.custom_op(
    "eqx_tace_7m::reciprocal_adjoint", mutates_args=(), device_types="cuda"
)
def reciprocal_adjoint(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    coeff: torch.Tensor,
    inverse: torch.Tensor,
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .tangent import projected_vjp, projection_capacity

    if not len(inverse):
        return projected_vjp(x, y, r, h, dh, src, dst, g, w, b, coeff, meta)
    dot = dh.new_empty((projection_capacity(len(dh)), w.size(1)))[: len(dh)]
    torch.mm(dh, w, out=dot)
    dx, dy, dr = torch.zeros_like(x), torch.zeros_like(y), x.new_zeros((len(y), 1))
    g = g.contiguous()
    dims = (x.size(1), w.size(1), y.size(1), g.size(1))
    plans = adjoint_plan(meta, dims, x.device, True, True, paired=True)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    for lo in range(0, len(src), TILE):
        sl = slice(lo, lo + TILE)
        n = len(src[sl])
        args = [
            v.data_ptr()
            for v in (x, y[sl], coeff, g, dx, dy[sl], dr[sl], src[sl], dst[sl])
        ]
        args.extend((n, dot.data_ptr(), inverse[sl].data_ptr()))
        runtime().launch(
            [(k, args, (n + 3) // 4, yz, 128, 0) for k, yz in plans], stream
        )
    return dx, dy, dr.to(r.dtype)


@reciprocal_adjoint.register_fake
def fake_adjoint(x, y, r, h, dh, src, dst, g, w, b, coeff, inverse, meta):
    return torch.empty_like(x), torch.empty_like(y), torch.empty_like(r)


def setup(ctx, inputs, output):
    x, y, r, h, dh, src, dst, _, w, b, inverse, ctx.meta = inputs
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(x, y, r, h, dh, src, dst, w, b, output[1], inverse)


def reverse(ctx, g, ignored):
    x, y, r, h, dh, src, dst, w, b, coeff, inverse = ctx.saved_tensors
    dx, dy, dr = reciprocal_adjoint(
        x, y, r, h, dh, src, dst, g, w, b, coeff, inverse, ctx.meta
    )
    return (dx, dy, dr) + (None,) * 9


reciprocal_projection.register_autograd(reverse, setup_context=setup)
