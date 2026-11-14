"""Exact analytic reference and fused sparse STF tensor products."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config
from ....common.irreducible_product import operands
from ....common.torch.cartesian import cartesian_product
from ....common.torch.literal_bilinear import evaluate as literal_product
from ....common.torch.sparse_bilinear import evaluate as sparse_product
from .. import symmetric as symmetric_kernel
from ..ictd import tensor_product as cartesian_kernel


def structured(module, x, y, cartesian_product, *, ordinary=False):
    """Apply separate trace branches in normalized symmetric storage."""
    value = cartesian_product(x, y, module.l1, module.l2, module.l3)
    if not module.symmetric:
        return value
    if module.packed:
        result = value @ module.expansion_0
        for t in range(1, len(module.coefficients)):
            value = cartesian_product(x, y, module.l1, module.l2, module.l3 - 2 * t)
            result = result + value @ getattr(module, f"expansion_{t}")
        return result.index_select(-1, module.indices)
    result = (
        symmetric_kernel.pack_reference(module.bases[0], value)
        if ordinary
        else module.bases[0].pack(value)
    )
    for t, basis in enumerate(module.bases):
        if t:
            value = cartesian_product(x, y, module.l1, module.l2, module.l3 - 2 * t)
            value = (
                symmetric_kernel.pack_reference(basis, value)
                if ordinary
                else basis.pack(value)
            )
            result = result + value @ getattr(module, f"expansion_{t}")
    basis = module.bases[0]
    return (
        symmetric_kernel.map_reference(result, basis.indices, basis.norm, True)
        if ordinary
        else basis.unpack(result)
    )


def reference(module, x, y):
    """The original delta/epsilon and dense expansion program."""
    return structured(module, x, y, cartesian_product, ordinary=True)


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
            dx = (symmetric @ y.unsqueeze(-1)).squeeze(-1) - trace[..., None] * y
            dy = (symmetric @ x.unsqueeze(-1)).squeeze(-1) - trace[..., None] * x
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
    if not config.TORCH_ICTP_PRODUCT or not getattr(module, "sparse_product", False):
        return structured(module, x, y, cartesian_product)
    if not x.is_cuda:
        if getattr(module, "literal_entries", ()) and (
            torch.compiler.is_compiling() or get_proxy_mode() is not None
        ):
            return literal_product(
                x, y, module.literal_dimensions, module.literal_entries
            )
        if module.l3 == 2 and module.l1 == module.l2 and module.l1 in (1, 2):
            if torch._C._are_functorch_transforms_active():
                return rank_two(x, y, module.l1)
            return RankTwoProduct.apply(x, y, module.l1)
        return structured(module, x, y, cartesian_product)
    result = sparse_product(x, y, operands(module))
    return result.index_select(-1, module.indices)


def evaluate(module, x, y, weight):
    """Retain compatible channel groups around fused angular products."""
    return cartesian_kernel.evaluate(module, x, y, weight)
