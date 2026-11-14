"""Native planar symmetric-traceless projection and compiler-visible reduction."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ... import config
from ...precision import validate_precision
from ...torch.co2 import basis as torch_kernel


def change_basis(module, features):
    """Apply compact fixed matrices through the shared native matrix primitive."""
    from ...common.cuda.grid import linear

    if not features.is_cuda:
        raise ValueError("backend='cuda' requires CUDA inputs.")
    if config.TORCH_REFERENCE:
        return torch_kernel.change_basis(module, features)
    values = []
    for mul, rank, dim, circular_dim, section in module.paths:
        matrix = getattr(module, f"basis_{rank}")
        value = features[..., section]
        if module.inverse:
            value = value.reshape(*features.shape[:-1], mul, dim)
            value = linear(value, matrix.T).transpose(-1, -2)
        else:
            value = value.reshape(*features.shape[:-1], circular_dim, mul)
            value = linear(value.transpose(-1, -2), matrix)
        values.append(value.flatten(-2))
    return torch.cat(values, -1) if values else features[..., :0]


@triton.jit
def planar_kernel(source, output, rows, RANK: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, 2**RANK)
    count = tl.full((2**RANK,), 0, tl.int32)
    for bit in tl.static_range(RANK):
        count += (column >> bit) & 1
    real = tl.where(count % 4 == 0, 1.0, tl.where(count % 4 == 2, -1.0, 0.0))
    imag = tl.where(count % 4 == 1, 1.0, tl.where(count % 4 == 3, -1.0, 0.0))
    index = row[:, None] * (2**RANK) + column[None, :]
    value = tl.load(source + index, row[:, None] < rows, other=0)
    a = tl.sum(value * real[None, :], 1)
    b = tl.sum(value * imag[None, :], 1)
    result = (a[:, None] * real[None, :] + b[:, None] * imag[None, :]) * 2.0 ** (
        1 - RANK
    )
    tl.store(output + index, result, row[:, None] < rows)


@triton_op("eqx::planar_project", mutates_args=())
def project_native(features: torch.Tensor, rank: int) -> torch.Tensor:
    """Project in one CUDA launch with an exact self-adjoint transpose."""
    validate_precision((features,))
    if not features.is_cuda or not 2 <= rank <= 8:
        raise ValueError("Native planar projection requires CUDA and rank 2..8.")
    if features.ndim < 1 or features.shape[-1] != 2**rank:
        raise ValueError("The trailing dimension does not match the tensor rank.")
    features = features.contiguous()
    result = torch.empty_like(features)
    rows = features.numel() // (2**rank)
    if rows:
        block = max(1, 512 // (2**rank))
        wrap_triton(planar_kernel)[(triton.cdiv(rows, block),)](
            features, result, rows, rank, block
        )
    return result


def project_context(ctx, inputs, output):
    ctx.rank = inputs[1]


def project_backward(ctx, gradient):
    return project_native(gradient, ctx.rank), None


project_native.register_autograd(project_backward, setup_context=project_context)


def planar_project(features, rank):
    """Preserve raw and nested forward AD through the exact Torch expression."""
    if forward_ad_active():
        return torch_kernel.planar_project(features, rank)
    return project_native(features, rank)


def project(module, features):
    """Apply native sparse paths or the selected bounded compiler schedule."""
    from equivariantx.ops.common.channel_product import project

    if not features.is_cuda:
        raise ValueError("backend='cuda' requires CUDA inputs.")
    if config.TORCH_REFERENCE:
        return torch_kernel.project_reference(module, features)
    if (
        config.PLANAR_PROJECTION_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
        and all(4 <= rank <= 8 for rank, _ in module.groups)
    ):
        return torch_kernel.project(module, features)
    if config.PLANAR_PROJECTION_NATIVE and all(
        2 <= rank <= 8 for rank, _ in module.groups
    ):
        return torch_kernel.project_blocks(
            module, features, lambda rank, value: planar_project(value, rank)
        )
    return project(module.kernel_metadata, features)
