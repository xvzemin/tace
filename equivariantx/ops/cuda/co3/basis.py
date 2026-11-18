"""CUDA Cartesian projection with compiler-visible rank-two kernels."""

import torch

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import evaluate as pointwise
from ...common.torch.basis import map_blocks, project_dense
from ...precision import validate_precision, validate_uniform
from ...torch.co3.basis import rank_two


@torch.library.custom_op("eqx::stf_coordinates", mutates_args=(), device_types="cuda")
def project_coordinates(
    x: torch.Tensor, columns: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Project normalized symmetric coordinates without a GEMM workspace."""
    validate_uniform((x, weights))
    if (
        columns.dtype != torch.long
        or columns.ndim != 2
        or columns.shape != weights.shape
        or x.size(-1) != columns.size(0)
    ):
        raise ValueError("The projection table must match the Cartesian coordinates.")
    x, columns, weights = (v.contiguous() for v in (x, columns, weights))
    result = torch.empty_like(x)
    n, k = columns.shape
    return pointwise(
        result,
        (x, columns, weights),
        f"""
        I row=i/{n}, c=i%{n}; T sum=0;
        #pragma unroll
        for(int j=0;j<{k};++j) {{
            I q=c*{k}+j; sum+=p0[row*{n}+p1[q]]*p2[q];
        }}
        out[i]=sum;
    """,
    )


@project_coordinates.register_fake
def coordinates_fake(x, columns, weights):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def coordinates_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def coordinates_backward(ctx, gradient):
    return project_coordinates(gradient, *ctx.saved_tensors), None, None


project_coordinates.register_autograd(
    coordinates_backward, setup_context=coordinates_context
)


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


def project(module, features, *, packed_degrees=()):
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
        if basis is not None:
            if degree not in packed_degrees:
                value = basis.pack(value)
            if hasattr(module, f"projector_{degree}") and not forward_ad_active():
                value = project_coordinates(
                    value,
                    getattr(module, f"columns_{degree}"),
                    getattr(module, f"projector_{degree}"),
                )
            else:
                value = (value @ matrix) @ matrix.T
            return basis.unpack(value)
        return project_dense(value, matrix, symmetric_basis=basis)

    return map_blocks(features, module.groups, apply)
