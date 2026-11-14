"""CUDA Cartesian projection with compiler-visible rank-two kernels."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ...common.torch.basis import map_blocks, project_dense
from ...precision import validate_precision
from ...torch.co3.basis import rank_two


def change_basis(module, features):
    """Apply fixed path matrices through the shared native matrix primitive."""
    from ...common.cuda.grid import linear

    if not features.is_cuda:
        raise ValueError("backend='cuda' requires CUDA inputs.")
    values = []
    for mul, degree, dim, section in getattr(module, "basis_groups", module.paths):
        matrix = getattr(module, f"basis_{degree}")
        value = features[..., section].reshape(*features.shape[:-1], mul, dim)
        values.append(linear(value, matrix.T if module.inverse else matrix).flatten(-2))
    if not values:
        return features[..., :0]
    return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


@triton.jit
def rank_two_kernel(source, result, size, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, component = index // 9, index % 9
    i, j = component // 3, component % 3
    valid = index < size
    a = tl.load(source + index, valid, other=0)
    b = tl.load(source + row * 9 + j * 3 + i, valid, other=0)
    diagonal = valid & (i == j)
    x = tl.load(source + row * 9, diagonal, other=0)
    y = tl.load(source + row * 9 + 4, diagonal, other=0)
    z = tl.load(source + row * 9 + 8, diagonal, other=0)
    value = (a + b) * 0.5 - (x + y + z) / 3.0
    tl.store(result + index, value, valid)


@triton_op("eqx::stf_rank_two", mutates_args=())
def project_rank_two(features: torch.Tensor) -> torch.Tensor:
    """Project contiguous rows in one CUDA launch; preserve FP32 or FP64."""
    validate_precision((features,))
    if not features.is_cuda:
        raise ValueError("The native projector requires CUDA features.")
    if features.ndim < 1 or features.shape[-1] != 9:
        raise ValueError("Rank-two Cartesian features must end in nine components.")
    features = features.contiguous()
    result = torch.empty_like(features)
    wrap_triton(rank_two_kernel)[(triton.cdiv(features.numel(), 256),)](
        features, result, features.numel(), 256
    )
    return result


def project_backward(ctx, gradient):
    return project_rank_two(gradient)


project_rank_two.register_autograd(project_backward)


def project(module, features):
    """Project low ranks with native kernels and higher ranks with CUDA GEMMs."""

    def apply(degree, value):
        if degree == 2:
            return rank_two(value) if forward_ad_active() else project_rank_two(value)
        matrix = getattr(module, f"basis_{degree}")
        basis = (
            module.symmetric_bases[str(degree)]
            if str(degree) in module.symmetric_bases
            else None
        )
        return project_dense(value, matrix, symmetric_basis=basis)

    return map_blocks(features, module.groups, apply)
