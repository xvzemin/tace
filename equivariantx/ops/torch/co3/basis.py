"""Cartesian basis conversion and symmetric traceless projection."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.torch.basis import map_blocks, project_dense
from . import symmetric as symmetric_kernel


def change_basis_reference(module, features):
    """Apply independent fixed basis matrices in ``mul_ir`` layout."""
    values = []
    for mul, degree, dim, section in module.paths:
        matrix = getattr(module, f"basis_{degree}")
        value = features[..., section].reshape(*features.shape[:-1], mul, dim)
        values.append((value @ (matrix if module.inverse else matrix.T)).flatten(-2))
    if not values:
        return features[..., :0]
    return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


def change_basis(module, features):
    """Share a GEMM across adjacent entries with the same fixed path matrix."""
    if config.TORCH_REFERENCE:
        return change_basis_reference(module, features)
    values = []
    for mul, degree, dim, section in getattr(module, "basis_groups", module.paths):
        matrix = getattr(module, f"basis_{degree}")
        value = features[..., section].reshape(*features.shape[:-1], mul, dim)
        values.append((value @ (matrix if module.inverse else matrix.T)).flatten(-2))
    if not values:
        return features[..., :0]
    return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


def rank_two(features):
    """Symmetrize and remove the trace without a dense projection matrix."""
    value = features.unflatten(-1, (3, 3))
    trace = value.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
    diagonal = torch.diag_embed(trace[..., None].expand(*trace.shape, 3))
    return ((value + value.transpose(-1, -2)) * 0.5 - diagonal).flatten(-2)


class RankTwo(torch.autograd.Function):
    """Self-adjoint linear projection without saved activation tensors."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features):
        return rank_two(features)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return RankTwo.apply(gradient)

    @staticmethod
    def jvp(ctx, tangent):
        return RankTwo.apply(tangent)


def project_reference(module, features):
    """Project with ordinary dense/factored basis matrices and standard AD."""

    def apply(degree, value):
        matrix = getattr(module, f"basis_{degree}")
        basis = (
            module.symmetric_bases[str(degree)]
            if str(degree) in module.symmetric_bases
            else None
        )
        if basis is None:
            return project_dense(value, matrix, square=degree == 2)
        value = symmetric_kernel.pack_reference(basis, value)
        value = (value @ matrix) @ matrix.T
        return symmetric_kernel.map_reference(value, basis.indices, basis.norm, True)

    return map_blocks(features, module.groups, apply)


def project(module, features):
    """Use analytic low-rank projection when the compiler can fuse its algebra."""
    if config.TORCH_REFERENCE or not config.TORCH_STF_PROJECTION:
        return project_reference(module, features)

    def apply(degree, value):
        if degree == 2 and (
            torch.compiler.is_compiling() or get_proxy_mode() is not None
        ):
            return (
                rank_two(value)
                if torch.compiler.is_compiling()
                else RankTwo.apply(value)
            )
        matrix = getattr(module, f"basis_{degree}")
        basis = (
            module.symmetric_bases[str(degree)]
            if str(degree) in module.symmetric_bases
            else None
        )
        return project_dense(value, matrix, square=degree == 2, symmetric_basis=basis)

    return map_blocks(features, module.groups, apply)
