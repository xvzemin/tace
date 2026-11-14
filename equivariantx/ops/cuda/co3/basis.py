"""CUDA Cartesian projection with compiler-visible rank-two kernels."""

import torch

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...common.torch.basis import map_blocks, project_dense
from ...precision import validate_precision
from ...torch.co3.basis import rank_two


def change_basis(module, features):
    """Apply fixed path matrices through the shared native matrix primitive."""
    from ...common.cuda.grid import linear

    if not features.is_cuda:
        raise ValueError("backend='cuda' requires CUDA inputs.")
    values = []
    for mul, degree, dim, section in module.basis_groups:
        matrix = getattr(module, f"basis_{degree}")
        value = features[..., section].reshape(*features.shape[:-1], mul, dim)
        values.append(linear(value, matrix.T if module.inverse else matrix).flatten(-2))
    if not values:
        return features[..., :0]
    return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


@torch.library.custom_op("eqx::stf_rank_two", mutates_args=(), device_types="cuda")
def project_rank_two(features: torch.Tensor) -> torch.Tensor:
    """Project rank-two Cartesian tensors onto their symmetric traceless part."""
    validate_precision((features,))
    if features.ndim < 1 or features.shape[-1] != 9:
        raise ValueError("Rank-two Cartesian tensors must end in nine components.")
    features = features.contiguous()
    result = torch.empty_like(features)
    return pointwise(
        result,
        (features,),
        """
        I row=i/9; int a=(i%9)/3,b=i%3;
        T value=(p0[i]+p0[row*9+b*3+a])*T(0.5);
        if(a==b) value-=(p0[row*9]+p0[row*9+4]+p0[row*9+8])/T(3);
        out[i]=value;
    """,
    )


@project_rank_two.register_fake
def fake(features):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


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
