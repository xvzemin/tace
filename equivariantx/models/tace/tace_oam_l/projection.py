"""Bounded radial-projection reuse in the frozen OAM-L force graph."""

import torch

from equivariantx.ops.common.cuda.o3.convolution import contract_direct

from .radial import RadialParameters, radial_heads
from .tangent_kernels import coordinate_adjoint, heads

BUCKET = 4096
TILE = 32768


def edge_limit(model, device):
    """Maximum number of saved edge rows per deep interaction."""
    bytes_per_row = 4 * sum(
        inter.edge_info.mlp[-1].out_dim for inter in model.interactions[1:]
    )
    budget = min(16 << 30, torch.cuda.get_device_properties(device).total_memory // 4)
    return budget // (BUCKET * bytes_per_row) * BUCKET


def saved_rows(edges, limit):
    return torch.sym_min(edges, torch.sym_max(2 * limit - edges, 0))


def padded_rows(rows):
    return (rows + BUCKET - 1) // BUCKET * BUCKET


@torch.library.custom_op(
    "eqx_oam_l_tangent::project", mutates_args=(), device_types="cuda"
)
def project(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    feats: torch.Tensor,
    fdot: torch.Tensor,
    st: torch.Tensor,
    tt: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    shape: torch.Tensor,
    packed: list[torch.Tensor],
    eps1: float,
    eps2: float,
    meta: str,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = y.shape[0]
    k = saved_rows(n, limit)
    cache = x.new_empty((padded_rows(k), packed[-1].shape[0]))
    if not k:
        z, d = torch.ops.eqx_oam_l_tangent.message(
            x, y, r, feats, fdot, st, tt, src, dst, shape, packed, eps1, eps2, meta
        )
        return z, d, cache
    p = RadialParameters(*packed, eps1, eps2)
    h, d = radial_heads(p, feats, st, tt)
    z = x.new_zeros(shape.shape)
    torch.addmm(p.projection_bias, h[:k], p.projection, out=cache[:k])
    scratch = x.new_empty((min(TILE, n - k), p.projection.shape[1]))
    for lo, hi in ((0, k), (k, n)):
        for start in range(lo, hi, TILE):
            sl = slice(start, min(start + TILE, hi))
            if start < k:
                rows = cache[sl]
            else:
                rows = scratch[: sl.stop - start]
                torch.addmm(p.projection_bias, h[sl], p.projection, out=rows)
            contract_direct(
                meta,
                src[sl],
                dst[sl],
                [((4,), (x, rows, rows[:0], y[sl], z), (z,), False)],
            )
    return z, d, cache


@project.register_fake
def project_shapes(
    x, y, r, feats, fdot, st, tt, src, dst, shape, packed, eps1, eps2, meta, limit
):
    rows = padded_rows(saved_rows(y.shape[0], limit))
    return (
        x.new_empty(shape.shape),
        x.new_empty((y.shape[0], 1)),
        x.new_empty((rows, packed[-1].shape[0])),
    )


@torch.library.custom_op(
    "eqx_oam_l_tangent::project_vjp", mutates_args=(), device_types="cuda"
)
def project_vjp(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    feats: torch.Tensor,
    fdot: torch.Tensor,
    st: torch.Tensor,
    tt: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    g: torch.Tensor,
    gd: torch.Tensor,
    cache: torch.Tensor,
    packed: list[torch.Tensor],
    eps1: float,
    eps2: float,
    meta: str,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = y.shape[0]
    k = saved_rows(n, limit)
    if not k:
        return torch.ops.eqx_oam_l_tangent.vjp(
            x, y, r, feats, fdot, st, tt, src, dst, g, gd, packed, eps1, eps2, meta
        )
    p = RadialParameters(*packed, eps1, eps2)
    dx, dy, dr = torch.zeros_like(x), torch.zeros_like(y), x.new_empty((n, 1))
    g = g.contiguous()
    scratch = x.new_empty((min(TILE, n - k), p.projection.shape[1]))
    deriv = x.new_empty((min(TILE, n), p.projection.shape[1]))
    for lo, hi in ((0, k), (k, n)):
        for start in range(lo, hi, TILE):
            sl = slice(start, min(start + TILE, hi))
            rows_count = sl.stop - start
            h, _, dh, dd = heads(p, feats[sl], fdot[sl], st[sl], tt[sl])
            if start < k:
                w = cache[sl]
            else:
                w = scratch[:rows_count]
                torch.addmm(p.projection_bias, h, p.projection, out=w)
            dw = deriv[:rows_count]
            torch.mm(dh, p.projection, out=dw)
            torch.mul(gd[sl], dd, out=dr[sl])
            coordinate_adjoint(
                meta, src[sl], dst[sl], x, y[sl], w, g, dx, dy[sl], dr[sl], dw
            )
    return dx, dy, dr.to(dtype=r.dtype)


@project_vjp.register_fake
def project_grad_shapes(
    x,
    y,
    r,
    feats,
    fdot,
    st,
    tt,
    src,
    dst,
    g,
    gd,
    cache,
    packed,
    eps1,
    eps2,
    meta,
    limit,
):
    return torch.empty_like(x), torch.empty_like(y), torch.empty_like(r)


def context(ctx, inputs, output):
    (
        x,
        y,
        r,
        feat,
        df,
        st,
        tt,
        src,
        dst,
        _,
        packed,
        ctx.eps1,
        ctx.eps2,
        ctx.meta,
        ctx.limit,
    ) = inputs
    ctx.mark_non_differentiable(output[2])
    ctx.save_for_backward(x, y, r, feat, df, st, tt, src, dst, output[2], *packed)


def reverse(ctx, g, gd, unused):
    x, y, r, feat, df, st, tt, src, dst, cache, *packed = ctx.saved_tensors
    dx, dy, dr = project_vjp(
        x,
        y,
        r,
        feat,
        df,
        st,
        tt,
        src,
        dst,
        g,
        gd,
        cache,
        packed,
        ctx.eps1,
        ctx.eps2,
        ctx.meta,
        ctx.limit,
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
        [None] * len(packed),
        None,
        None,
        None,
        None,
    )


project.register_autograd(reverse, setup_context=context)
