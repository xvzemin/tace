"""Node-tiled quadratic product and its frozen coefficient contraction."""

import torch

from equivariantx.utils.metadata import parse_metadata

from .product_kernels import square

TILE = 2048
FORWARD = "(((0, 0, 1), 2, 0, 0),)"
ADJOINT = "(((0, 0, 1), 0, 0, 4), ((0, 0, 1), 1, 0, 4))"


def _slab(cg, linear, x, z, w, order, inverse, dy=None):
    qin = x.index_select(1, order[0])
    _, width, *_ = parse_metadata(linear)
    tmp = x.new_empty((0, width))
    if dy is None:
        q = torch.ops.eqx.local_channel_product(cg, FORWARD, [qin, x.new_empty((0, inverse[1].numel()))])[0]
        q = q.index_select(1, inverse[1])
        return torch.ops.eqx.element_linear(linear, 2, z, q, w.unsqueeze(1), tmp)
    dq = torch.ops.eqx.element_linear(linear, 2, z, dy, w.unsqueeze(1), tmp)
    dq = dq.index_select(1, order[1])
    dx = torch.ops.eqx.local_channel_product(cg, ADJOINT, [qin, dq])[0]
    return dx.index_select(1, inverse[0])


@torch.library.custom_op("eqx_oam_l_product::quadratic", mutates_args=(), device_types="cuda")
def _quadratic(cg: str, linear: str, reverse: str, x: torch.Tensor, z: torch.Tensor,
               w: torch.Tensor, wt: torch.Tensor, perm: list[torch.Tensor]) -> torch.Tensor:
    result = x.new_empty((x.shape[0], parse_metadata(linear)[1]))
    for i in range(0, x.shape[0], TILE):
        s = slice(i, i + TILE)
        result[s].copy_(_slab(cg, linear, x[s], z[s], w, perm[:2], perm[2:]))
    return result


@_quadratic.register_fake
def _fake(cg, linear, reverse, x, z, w, wt, perm):
    return x.new_empty((x.shape[0], parse_metadata(linear)[1]))


@torch.library.custom_op("eqx_oam_l_product::quadratic_vjp", mutates_args=(), device_types="cuda")
def _vjp(cg: str, reverse: str, x: torch.Tensor, z: torch.Tensor, wt: torch.Tensor,
         perm: list[torch.Tensor], dy: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for i in range(0, x.shape[0], TILE):
        s = slice(i, i + TILE)
        result[s].copy_(_slab(cg, reverse, x[s], z[s], wt, perm[:2], perm[2:], dy[s]))
    return result


@_vjp.register_fake
def _fake_vjp(cg, reverse, x, z, wt, perm, dy):
    return torch.empty_like(x)


def _context(ctx, inputs, output):
    cg, _, reverse, x, z, _, wt, perm = inputs
    ctx.cg, ctx.reverse = cg, reverse
    ctx.save_for_backward(x, z, wt, *perm)


def _backward(ctx, dy):
    x, z, wt, *perm = ctx.saved_tensors
    if torch.is_grad_enabled():
        dx = _slab(ctx.cg, ctx.reverse, x, z, wt, perm[:2], perm[2:], dy)
    else:
        dx = _vjp(ctx.cg, ctx.reverse, x, z, wt, perm, dy)
    return None, None, None, dx, None, None, None, [None]*len(perm)


_quadratic.register_autograd(_backward, setup_context=_context)


def frozen_quadratic(product, x, z):
    """Apply all quadratic paths and their element coefficients in node tiles."""
    coef = product.coefs[1]
    fwd, rev, _, _, blocks = coef.frozen_plan
    w = coef.weight
    wt = torch.cat([w[:,o:o+i*j].reshape(w.shape[0],i,j).transpose(1,2).reshape(w.shape[0], -1)
                    for o,i,j in blocks],1)
    perm = [product.input_order, product.output_order, product.input_inverse, product.output_inverse]
    y = _quadratic(product.cg_metadata, fwd, rev, x, z, w, wt, perm)
    if coef.bias is not None:
        y = y + coef.bias[z].index_select(1,coef._bias_index) * coef._bias_mask
    return y
