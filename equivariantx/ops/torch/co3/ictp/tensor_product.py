"""Exact analytic reference and fused sparse STF tensor products."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.autograd import forward_ad_active
from .... import config
from ....common.irreducible_product import operands
from ....common.torch.cartesian import cartesian_product
from ....common.torch.cartesian import evaluate as optimized_product
from ....common.torch.literal_bilinear import evaluate as literal_product
from ....common.torch.sparse_bilinear import evaluate as sparse_product
from .. import symmetric as symmetric_kernel
from ..channels import grouped

AXIAL_WORKSPACE_BYTES = 16 << 20
CONTRACTION_WORKSPACE_BYTES = 8 << 20


def structured(module, x, y, cartesian_product, *, ordinary=False, pack_product=None):
    """Apply separate trace branches in normalized symmetric storage."""
    if module.path_type == "direct":
        return cartesian_product(x, y, module.l1, module.l2, module.l3)
    if module.path_type == "symmetrize":
        return permutation_mean(
            module, cartesian_product(x, y, module.l1, module.l2, module.l3)
        )
    if module.packed:
        value = cartesian_product(x, y, module.l1, module.l2, module.l3)
        result = value @ module.expansion_0
        for t in range(1, len(module.coefficients)):
            value = cartesian_product(x, y, module.l1, module.l2, module.l3 - 2 * t)
            result = result + value @ getattr(module, f"expansion_{t}")
        return result.index_select(-1, module.indices)
    for t, basis in enumerate(module.bases):
        rank = module.l3 - 2 * t
        if pack_product is not None and rank >= 4:
            value = pack_product(x, y, module.l1, module.l2, basis)
        else:
            value = cartesian_product(x, y, module.l1, module.l2, rank)
            value = (
                symmetric_kernel.pack_reference(basis, value)
                if ordinary
                else basis.pack(value)
            )
        if t:
            result = result + value @ getattr(module, f"expansion_{t}")
        else:
            result = value
    basis = module.bases[0]
    return (
        symmetric_kernel.map_reference(result, basis.indices, basis.norm, True)
        if ordinary
        else basis.unpack(result)
    )


def reference(module, x, y):
    """The original delta/epsilon and dense expansion program."""
    return structured(module, x, y, cartesian_product, ordinary=True)


def permutation_mean(module, raw, scale=1.0):
    size = module.counts.numel()
    sums = raw.new_zeros((*raw.shape[:-1], size)).index_add(-1, module.indices, raw)
    means = sums / module.counts if scale == 1.0 else sums * (scale / module.counts)
    return means.index_select(-1, module.indices)


def axial_product(module, x, y):
    """Contract saturated epsilon paths with a cross product over the summed indices."""
    k = 3**module.contractions
    free = 3 ** (module.l3 - 1)
    if k == 1:
        if module.l1 >= module.l2:
            a, b = x.unflatten(-1, (free, 3)), y.unsqueeze(-2)
        else:
            a, b = x.unsqueeze(-2), y.unflatten(-1, (3, free)).transpose(-1, -2)
        a, b = torch.broadcast_tensors(a, b)
        raw = torch.linalg.cross(a, b, dim=-1)
    elif (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or math.prod(torch.broadcast_shapes(x.shape[:-1], y.shape[:-1]))
        * 3
        * free
        * k
        * x.element_size()
        <= AXIAL_WORKSPACE_BYTES
    ):
        if module.l1 >= module.l2:
            a = x.unflatten(-1, (free, 3, k)).transpose(-1, -2)
            b = y.unflatten(-1, (k, 3)).unsqueeze(-3)
        else:
            a = x.unflatten(-1, (3, k)).transpose(-1, -2).unsqueeze(-3)
            b = y.unflatten(-1, (k, 3, free)).movedim(-1, -3)
        a, b = torch.broadcast_tensors(a, b)
        raw = torch.linalg.cross(a, b, dim=-1).sum(-2)
    else:
        if module.l1 >= module.l2:
            gram = (x.unflatten(-1, (free * 3, k)) @ y.unflatten(-1, (k, 3))).unflatten(
                -2, (free, 3)
            )
        else:
            gram = (
                (x.unflatten(-1, (3, k)) @ y.unflatten(-1, (k, 3 * free)))
                .unflatten(-1, (3, free))
                .movedim(-1, -3)
            )
        raw = torch.stack(
            [gram[..., a, b] - gram[..., b, a] for a, b in ((1, 2), (2, 0), (0, 1))], -1
        )
    scale = (2 * k) ** -0.5
    if module.l3 == 1:
        return raw.squeeze(-2) * scale
    if module.l3 == 2:
        return ((raw + raw.transpose(-1, -2)) * (0.5 * scale)).flatten(-2)
    return permutation_mean(module, raw.flatten(-2), scale)


def rank_two(x, y, rank):
    """Direct rank-two symmetrization with no expansion-matrix products."""
    if rank == 1:
        product = x.unsqueeze(-1) * y.unsqueeze(-2)
    else:
        product = x.unflatten(-1, (3, 3)) @ y.unflatten(-1, (3, 3))
    trace = (x * y).sum(-1) / 3
    diagonal = torch.diag_embed(trace[..., None].expand(*trace.shape, 3))
    return (
        ((product + product.transpose(-1, -2)) * 0.5 - diagonal)
        / math.sqrt(3 ** (rank - 1))
    ).flatten(-2)


@torch.compiler.allow_in_graph
class RankTwoProduct(torch.autograd.Function):
    """Bilinear adjoints retaining only operands, with ordinary higher-order AD."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x, y, rank):
        return rank_two(x, y, rank)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, ctx.rank = inputs
        ctx.save_for_backward(x, y)
        ctx.save_for_forward(x, y)

    @staticmethod
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        matrix = gradient.unflatten(-1, (3, 3)) / math.sqrt(3 ** (ctx.rank - 1))
        trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
        symmetric = (matrix + matrix.transpose(-1, -2)) * 0.5
        if ctx.rank == 1:
            dx = (symmetric * y.unsqueeze(-2)).sum(-1) - trace[..., None] * y
            dy = (symmetric * x.unsqueeze(-2)).sum(-1) - trace[..., None] * x
        else:
            dx = (symmetric @ y.unflatten(-1, (3, 3)).transpose(-1, -2)).flatten(
                -2
            ) - trace[..., None] * y
            dy = (x.unflatten(-1, (3, 3)).transpose(-1, -2) @ symmetric).flatten(
                -2
            ) - trace[..., None] * x
        return dx.sum_to_size(x.shape), dy.sum_to_size(y.shape), None

    @staticmethod
    def jvp(ctx, dx, dy, unused_rank):
        x, y = ctx.saved_tensors
        dx = torch.zeros_like(x) if dx is None else dx
        dy = torch.zeros_like(y) if dy is None else dy
        return rank_two(dx, y, ctx.rank) + rank_two(x, dy, ctx.rank)


def angular(module, x, y):
    if config.TORCH_REFERENCE:
        return reference(module, x, y)
    if config.TORCH_ICTP_PRODUCT and (
        module.path_type == "symmetrize" or (module.l3 == 1 and module.l1 == module.l2)
    ):
        return axial_product(module, x, y)
    if (
        config.TORCH_ICTP_PRODUCT
        and module.path_type == "direct"
        and module.contractions > 0
        and module.l3 > 0
        and x.is_cuda
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
        and math.prod(torch.broadcast_shapes(x.shape[:-1], y.shape[:-1]))
        * 3 ** max(module.l1, module.l2)
        * x.element_size()
        <= CONTRACTION_WORKSPACE_BYTES
    ):
        k = 3**module.contractions
        free = 3**module.l3
        if module.l1 > module.l2:
            return (x.unflatten(-1, (free, k)) * y.unsqueeze(-2)).sum(-1) * (k**-0.5)
        return (x.unsqueeze(-1) * y.unflatten(-1, (k, free))).sum(-2) * (k**-0.5)
    if not config.TORCH_ICTP_PRODUCT or not getattr(module, "sparse_product", False):
        if (
            x.is_cuda
            and hasattr(module, "symmetric_products")
            and not (torch.compiler.is_compiling() or get_proxy_mode() is not None)
        ):
            return structured(
                module,
                x,
                y,
                optimized_product,
                pack_product=lambda x, y, l1, l2, b: module.symmetric_products[
                    str(b.rank)
                ](x, y, b.norm),
            )
        return structured(module, x, y, optimized_product)
    if not x.is_cuda:
        if getattr(module, "literal_entries", ()) and (
            torch.compiler.is_compiling() or get_proxy_mode() is not None
        ):
            return literal_product(
                x, y, module.literal_dimensions, module.literal_entries
            )
    if module.l3 == 2 and module.l1 == module.l2 and module.l1 in (1, 2):
        if forward_ad_active():
            return rank_two(x, y, module.l1)
        return RankTwoProduct.apply(x, y, module.l1)
    if not x.is_cuda:
        return structured(module, x, y, cartesian_product)
    result = sparse_product(x, y, operands(module))
    return result.index_select(-1, module.indices)


def evaluate(module, x, y, weight):
    """Retain compatible channel groups around fused angular products."""
    if module.backend == "torch" and config.TORCH_REFERENCE:
        zero = x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
        return module.reference(x, y, weight, module.path_scales) + zero
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
    return grouped(module, x, y, weight, shape)
