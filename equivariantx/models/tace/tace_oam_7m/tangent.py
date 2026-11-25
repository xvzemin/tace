"""Frozen distance-tangent evaluation of the two-hidden-layer 7M radial heads."""

from contextvars import ContextVar
from functools import lru_cache

import torch

from equivariantx.models.tace.tace_oam_l.convolution import _affine
from equivariantx.models.tace.tace_oam_l.tangent import embedding, radial_embedding
from equivariantx.models.tace.tace_oam_l.tangent_kernels import coordinate_adjoint
from equivariantx.ops.common.cuda.o3.convolution import contract_direct
from equivariantx.ops.common.cuda.pointwise import launch
from equivariantx.utils.metadata import compact_metadata

from .model import SevenRepresentation, radial_parameters
from .product import FrozenBilinearProduct
from .radial_batch import batch_radial
from .reciprocal import reciprocal_heads, reciprocal_projection


@lru_cache(maxsize=16)
def norm_source(first, dual, eps):
    width = 128 if first else 64
    return f"""
    using I = long long;
    __device__ float sum(float x) {{
        #pragma unroll
        for (int d=16;d;d/=2) x+=__shfl_xor_sync(0xffffffff,x,d);
        return x;
    }}
    extern "C" __global__ void run(
      float* p, float* dp, const float* src, const float* dst, const I* s, const I* t,
      const float* gain, const float* shift, I rows) {{
      int lane=threadIdx.x%32;
      for(I i=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32;i<rows*(I){width // 64};
          i+=I(gridDim.x)*blockDim.x/32) {{
        I row=i/{width // 64}; int c=(i%{width // 64})*64+lane;
        I off=row*{width}+c;
        float a=p[off], b=p[off+32];
        if ({int(first)}) {{
            a=(a+dst[t[row]*128+c])+src[s[row]*128+c];
            b=(b+dst[t[row]*128+c+32])+src[s[row]*128+c+32];
        }}
        float av=sum(a+b)/64.f;
        a-=av; b-=av;
        float inv=rsqrtf(sum(a*a+b*b)/64.f+float({float(eps).hex()}));
        a*=inv; b*=inv;
        float za=a*gain[c]+shift[c], zb=b*gain[c+32]+shift[c+32];
        float sa=1.f/(1.f+expf(-za)), sb=1.f/(1.f+expf(-zb));
        p[off]=za*sa; p[off+32]=zb*sb;
        if ({int(dual)}) {{
           float u=dp[off],v=dp[off+32];
           float m=sum(u+v)/64.f, cov=sum(u*a+v*b)/64.f;
           dp[off]=(u-m-a*cov)*inv*gain[c]*sa*(1.f+za*(1.f-sa));
           dp[off+32]=(v-m-b*cov)*inv*gain[c+32]*sb*(1.f+zb*(1.f-sb));
        }}
      }}
    }}"""


def pack(inter, update):
    i, d = inter.edge_info.mlp, inter.scatter_norm.edge_density.mlp
    return [
        q.contiguous()
        for q in (
            *radial_parameters(inter, update),
            torch.cat((i[1].weight, d[1].weight)),
            torch.cat((i[1].bias, d[1].bias)),
            i[3].get_weight(),
            i[3].bias,
            i[4].weight,
            i[4].bias,
            d[-1].get_weight(),
            d[-1].bias,
            i[-1].get_weight(),
            i[-1].bias,
        )
    ]


def hidden(feat, tangent, source, target, p, eps1, eps2, dual):
    edge, src, dst, bias, gain, shift, w, b, g2, s2, dw, db, _, _ = p
    pre = torch.addmm(bias, feat, edge)
    dt = tangent @ edge if dual else pre
    n = len(feat)
    launch(
        norm_source(True, dual, eps1),
        [pre, dt, src, dst, source, target, gain, shift],
        (n,),
        n * 64,
        threads=128,
    )
    mid = torch.addmm(b, pre[:, :64], w)
    dm = dt[:, :64] @ w if dual else mid
    launch(
        norm_source(False, dual, eps2),
        [mid, dm, src, dst, source, target, g2, s2],
        (n,),
        n * 32,
        threads=128,
    )
    raw = torch.addmm(db, pre[:, 64:], dw)
    d = raw.square().tanh()
    dd = (dt[:, 64:] @ dw) * (2.0 * raw) * (1.0 - d.square()) if dual else d
    return mid, d, dm, dd


@torch.library.custom_op("eqx_tace_7m::message", mutates_args=(), device_types="cuda")
def radial_message(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    feat: torch.Tensor,
    tangent: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    shape: torch.Tensor,
    p: list[torch.Tensor],
    eps1: float,
    eps2: float,
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, d, _, _ = hidden(feat, tangent, s, t, p, eps1, eps2, False)
    return _affine(x, y, h, p[-2], p[-1], src, dst, shape, meta), d


@radial_message.register_fake
def fake_message(x, y, r, feat, tangent, s, t, src, dst, shape, p, eps1, eps2, meta):
    return x.new_empty(shape.shape), x.new_empty((y.shape[0], 1))


@torch.library.custom_op(
    "eqx_tace_7m::message_vjp", mutates_args=(), device_types="cuda"
)
def message_vjp(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    feat: torch.Tensor,
    tangent: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    g: torch.Tensor,
    gd: torch.Tensor,
    p: list[torch.Tensor],
    eps1: float,
    eps2: float,
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, dy = torch.zeros_like(x), torch.zeros_like(y)
    dr = x.new_empty((len(y), 1))
    g = g.contiguous()
    tile = 32768
    w, b = p[-2:]
    weight, dw = (x.new_empty(min(tile, len(y)), w.shape[1]) for _ in range(2))
    for start in range(0, len(y), tile):
        sl = slice(start, start + tile)
        h, _, dh, dd = hidden(feat[sl], tangent[sl], s[sl], t[sl], p, eps1, eps2, True)
        rows, deriv = weight[: len(h)], dw[: len(h)]
        torch.addmm(b, h, w, out=rows)
        torch.mm(dh, w, out=deriv)
        torch.mul(dd, gd[sl], out=dr[sl])
        coordinate_adjoint(
            meta, src[sl], dst[sl], x, y[sl], rows, g, dx, dy[sl], dr[sl], deriv
        )
    return dx, dy, dr.to(r.dtype)


@message_vjp.register_fake
def fake_vjp(x, y, r, feat, tangent, s, t, src, dst, g, gd, p, eps1, eps2, meta):
    return torch.empty_like(x), torch.empty_like(y), torch.empty_like(r)


def setup(ctx, inputs, output):
    x, y, r, feat, tangent, s, t, src, dst, _, p, ctx.eps1, ctx.eps2, ctx.meta = inputs
    ctx.save_for_backward(x, y, r, feat, tangent, s, t, src, dst, *p)


def reverse(ctx, g, gd):
    x, y, r, feat, tangent, s, t, src, dst, *p = ctx.saved_tensors
    dx, dy, dr = message_vjp(
        x, y, r, feat, tangent, s, t, src, dst, g, gd, p, ctx.eps1, ctx.eps2, ctx.meta
    )
    return (
        dx,
        dy,
        dr,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        [None] * len(p),
        None,
        None,
        None,
    )


radial_message.register_autograd(reverse, setup_context=setup)


def tangent_embedding(rep):
    for inter in rep.interactions:
        m, d = inter.edge_info.mlp, inter.scatter_norm.edge_density.mlp
        if not (
            isinstance(m[2], torch.nn.SiLU)
            and isinstance(d[2], torch.nn.SiLU)
            and isinstance(m[5], torch.nn.SiLU)
            and isinstance(m[1], torch.nn.LayerNorm)
            and isinstance(d[1], torch.nn.LayerNorm)
            and isinstance(m[4], torch.nn.LayerNorm)
            and m[1].eps == d[1].eps
        ):
            return None
    return radial_embedding(rep)


def distance_inputs(rep, length, matrix=None):
    basis = rep.radial_basis.radial_fn
    if matrix is None:
        matrix = rep.edge_embedding.radial_proj(
            torch.eye(8, device=length.device, dtype=torch.float32)
        ).contiguous()
    feat, dt = embedding(
        length.detach(),
        basis.bessel_weights,
        basis.prefactor,
        matrix,
        rep.edge_embedding.act1.acts[0].cst,
    )
    return feat, dt


@torch.library.custom_op("eqx_tace_7m::encode", mutates_args=(), device_types="cuda")
def encode(
    feat: torch.Tensor,
    tangent: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    p: list[torch.Tensor],
    eps1: float,
    eps2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return hidden(feat, tangent, s, t, p, eps1, eps2, True)


@encode.register_fake
def fake_encode(feat, tangent, s, t, p, eps1, eps2):
    n = feat.shape[0]
    return tuple(feat.new_empty((n, c)) for c in (64, 1, 64, 1))


@torch.library.custom_op("eqx_tace_7m::encoded", mutates_args=(), device_types="cuda")
def encoded(
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
    meta: str,
) -> torch.Tensor:
    return _affine(x, y, h, w, b, src, dst, shape, meta)


@encoded.register_fake
def fake_encoded(x, y, r, h, dh, src, dst, shape, w, b, meta):
    return x.new_empty(shape.shape)


@torch.library.custom_op(
    "eqx_tace_7m::encoded_vjp", mutates_args=(), device_types="cuda"
)
def encoded_vjp(
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
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return projection_adjoint(x, y, r, h, dh, src, dst, g, w, b, meta)


def projection_adjoint(x, y, r, h, dh, src, dst, g, w, b, meta, coeff=None):
    dx, dy = torch.zeros_like(x), torch.zeros_like(y)
    dr = x.new_zeros((len(y), 1))
    g = g.contiguous()
    tile = 32768
    dots = x.new_empty(min(tile, len(y)), w.shape[1])
    values = torch.empty_like(dots) if coeff is None else None
    for start in range(0, len(y), tile):
        sl = slice(start, start + tile)
        deriv = dots[: len(h[sl])]
        rows = (
            coeff[sl]
            if coeff is not None
            else torch.addmm(b, h[sl], w, out=values[: len(h[sl])])
        )
        torch.mm(dh[sl], w, out=deriv)
        coordinate_adjoint(
            meta, src[sl], dst[sl], x, y[sl], rows, g, dx, dy[sl], dr[sl], deriv
        )
    return dx, dy, dr.to(r.dtype)


@encoded_vjp.register_fake
def fake_encoded_vjp(x, y, r, h, dh, src, dst, g, w, b, meta):
    return torch.empty_like(x), torch.empty_like(y), torch.empty_like(r)


def encoded_setup(ctx, inputs, output):
    x, y, r, h, dh, src, dst, _, w, b, ctx.meta = inputs
    ctx.save_for_backward(x, y, r, h, dh, src, dst, w, b)


def encoded_reverse(ctx, g):
    x, y, r, h, dh, src, dst, w, b = ctx.saved_tensors
    dx, dy, dr = encoded_vjp(x, y, r, h, dh, src, dst, g, w, b, ctx.meta)
    return dx, dy, dr, None, None, None, None, None, None, None, None


encoded.register_autograd(encoded_reverse, setup_context=encoded_setup)


retain_projections = ContextVar("eqx_7m_retain_projections", default=False)


def projection_capacity(edges):
    """Round a projected edge workspace to a reusable allocation bucket."""
    step = 1 << max(0, (edges - 1).bit_length() - 4)
    return ((edges + step - 1) // step) * step


@torch.library.custom_op("eqx_tace_7m::projected", mutates_args=(), device_types="cuda")
def projected(
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
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not retain_projections.get():
        return _affine(x, y, h, w, b, src, dst, shape, meta), h.new_empty(
            (0, w.shape[1])
        )
    z = x.new_zeros(shape.shape)
    coeff = h.new_empty((projection_capacity(len(h)), w.shape[1]))[: len(h)]
    for lo in range(0, len(h), 32768):
        sl = slice(lo, lo + 32768)
        c = torch.addmm(b, h[sl], w, out=coeff[sl])
        contract_direct(
            meta, src[sl], dst[sl], [((4,), (x, c, c[:0], y[sl], z), (z,), False)]
        )
    return z, coeff


@projected.register_fake
def projected_shape(x, y, r, h, dh, src, dst, shape, w, b, meta):
    saved = torch.library.get_ctx().new_dynamic_size()
    return x.new_empty(shape.shape), h.new_empty((saved, w.shape[1]))


@torch.library.custom_op(
    "eqx_tace_7m::projected_vjp", mutates_args=(), device_types="cuda"
)
def projected_vjp(
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
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return projection_adjoint(
        x, y, r, h, dh, src, dst, g, w, b, meta, coeff if len(coeff) else None
    )


@projected_vjp.register_fake
def projected_vjp_shape(x, y, r, h, dh, src, dst, g, w, b, coeff, meta):
    return torch.empty_like(x), torch.empty_like(y), torch.empty_like(r)


def projected_setup(ctx, inputs, output):
    x, y, r, h, dh, src, dst, _, w, b, ctx.meta = inputs
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(x, y, r, h, dh, src, dst, w, b, output[1])


def projected_reverse(ctx, g, ignored):
    x, y, r, h, dh, src, dst, w, b, coeff = ctx.saved_tensors
    dx, dy, dr = projected_vjp(x, y, r, h, dh, src, dst, g, w, b, coeff, ctx.meta)
    return dx, dy, dr, None, None, None, None, None, None, None, None


projected.register_autograd(projected_reverse, setup_context=projected_setup)


def tangent_head(inter, edge):
    m = inter.edge_info.mlp
    return (
        pack(inter, edge),
        compact_metadata(inter.rejector.eqx_tp.kernel_metadata),
        m[1].eps,
        m[4].eps,
    )


def tangent_message(
    inter,
    edge,
    x,
    attrs,
    r,
    feat,
    dt,
    cutoff,
    index,
    n,
    s,
    t,
    save_hidden=False,
    head=None,
):
    conv = inter.rejector
    x = conv.reshape_in(x).contiguous()
    norm = inter.scatter_norm
    src, dst = index[0].contiguous(), index[1].contiguous()
    shape = x.new_empty(1).expand(n, conv.eqx_tp.output_dim)
    p, meta, eps1, eps2 = tangent_head(inter, edge) if head is None else head
    if save_hidden:
        h, d, dh, dd = encode(feat, dt, s, t, p, eps1, eps2)
        d = d + (r - r.detach()).to(d.dtype) * dd
        z = encoded(
            x, attrs.contiguous(), r, h, dh, src, dst, shape, p[-2], p[-1], meta
        )
    else:
        z, d = radial_message(
            x,
            attrs.contiguous(),
            r,
            feat,
            dt,
            s,
            t,
            src,
            dst,
            shape,
            p,
            eps1,
            eps2,
            meta,
        )
    denom = d.new_zeros(n, 1).index_add_(0, index[1], d * cutoff)
    return inter.linear_down(conv.reshape_out(z)) / (norm.alpha + norm.beta * denom)


class SevenDistanceTangent(SevenRepresentation):
    """Evaluate frozen 7M interactions and their first coordinate derivatives."""

    save_hidden = False

    def edge_geometry(self, vector, length, types, index):
        if length.dtype != torch.float64 or self.model_dtype != torch.float32:
            raise ValueError(
                "The frozen 7M distance tangent requires FP64 geometry and FP32 weights."
            )
        feat, dt = distance_inputs(self, length)
        cutoff = self.radial_basis.cutoff_fn(length).to(self.model_dtype)
        attrs = self.o3_angular_basis(vector / length.clamp_min(1e-12)).to(
            self.model_dtype
        )
        return attrs * cutoff, length, feat, dt, cutoff

    def edge_message(self, inter, edge, features, geometry, index, n, src, dst):
        return tangent_message(
            inter, edge, features, *geometry, index, n, src, dst, self.save_hidden
        )


class BatchedSeven(SevenDistanceTangent):
    """Evaluate all five frozen radial heads together before message passing."""

    save_projection = False
    reciprocal = False

    def prepare(self):
        for layer in self.products:
            layer.__class__ = FrozenBilinearProduct
            layer.prepare()

    def edge_geometry(self, vector, length, types, index):
        y, r, f, dt, cut = super().edge_geometry(vector, length, types, index)
        pars = [pack(i, e) for i, e in zip(self.interactions, self.edge_updates)]
        p = [
            torch.cat([v[0] for v in pars], 1).contiguous(),
            *(torch.stack([v[i] for v in pars]).contiguous() for i in range(1, 12)),
        ]
        layer = self.interactions[0].edge_info.mlp
        if self.reciprocal:
            h, d, dh, dd, inverse = reciprocal_heads(
                f, dt, index[0], index[1], types, p, layer[1].eps, layer[4].eps
            )
            return y, r, h, d, dh, dd, cut, inverse
        h, d, dh, dd = batch_radial(
            f, dt, types[index[0]], types[index[1]], p, layer[1].eps, layer[4].eps
        )
        return y, r, h, d, dh, dd, cut

    def edge_message(self, inter, edge, features, geometry, index, n, src, dst):
        y, r, h, d, dh, dd, cut = geometry[:7]
        i = inter.layer
        d = d[i] + (r - r.detach()).to(d.dtype) * dd[i]
        c = inter.rejector
        x = c.reshape_in(features).contiguous()
        w = inter.edge_info.mlp[-1]
        args = (
            x,
            y.contiguous(),
            r,
            h[i],
            dh[i],
            index[0].contiguous(),
            index[1].contiguous(),
            x.new_empty(1).expand(n, c.eqx_tp.output_dim),
            w.get_weight().contiguous(),
            w.bias.contiguous(),
            compact_metadata(c.eqx_tp.kernel_metadata),
        )
        if self.reciprocal:
            z = reciprocal_projection(*args[:-1], geometry[7], args[-1])[0]
        else:
            z = projected(*args)[0] if self.save_projection else encoded(*args)
        norm = inter.scatter_norm
        denom = d.new_zeros((n, 1)).index_add_(0, index[1], d * cut)
        return inter.linear_down(c.reshape_out(z)) / (norm.alpha + norm.beta * denom)
