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
    for mul, degree, dim, section in module.basis_groups:
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


def rank_two_layout(features, sections):
    """Torch decomposition of a mixed-layout rank-two projection."""
    pieces, end = [], 0
    for start, stop in zip(sections[::2], sections[1::2], strict=True):
        pieces.append(features[..., end:start])
        values = features[..., start:stop].unflatten(-1, ((stop - start) // 9, 9))
        pieces.append(rank_two(values).flatten(-2))
        end = stop
    pieces.append(features[..., end:])
    return torch.cat(pieces, dim=-1)


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


class ParityProjection(torch.autograd.Function):
    """Self-adjoint projection of normalized symmetric coordinates."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value, columns, weights):
        return (value[..., columns] * weights).sum(-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[1:])
        ctx.save_for_forward(*inputs[1:])

    @staticmethod
    def backward(ctx, gradient):
        return ParityProjection.apply(gradient, *ctx.saved_tensors), None, None

    @staticmethod
    def jvp(ctx, tangent, dc, dw):
        return ParityProjection.apply(tangent, *ctx.saved_tensors)


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


def project(module, features, *, packed_degrees=()):
    """Use analytic low-rank projection when the compiler can fuse its algebra."""
    if not packed_degrees and (
        config.TORCH_REFERENCE or not config.TORCH_STF_PROJECTION
    ):
        return project_reference(module, features)

    def apply(degree, value):
        if degree == 2 and (
            value.is_cuda
            or torch.compiler.is_compiling()
            or get_proxy_mode() is not None
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
        if basis is None:
            return project_dense(value, matrix, square=degree == 2)
        if degree not in packed_degrees:
            value = symmetric_kernel.pack(basis, value)
        if (
            value.is_cuda
            and hasattr(module, f"projector_{degree}")
            and not torch.compiler.is_compiling()
            and get_proxy_mode() is None
            and value.numel()
            * value.element_size()
            * getattr(module, f"columns_{degree}").shape[1]
            <= 8 << 20
        ):
            columns = getattr(module, f"columns_{degree}")
            weights = getattr(module, f"projector_{degree}")
            value = ParityProjection.apply(value, columns, weights)
        else:
            value = (value @ matrix) @ matrix.T
        return symmetric_kernel.unpack(basis, value)

    return map_blocks(features, module.groups, apply)
