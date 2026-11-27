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


@torch.library.custom_op(
    "eqx::stf_rank_two_layout", mutates_args=(), device_types="cuda"
)
def project_rank_two_layout(
    features: torch.Tensor, sections: list[int]
) -> torch.Tensor:
    """Project selected nine-component blocks in one mixed-layout pass."""
    validate_precision((features,))
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Rank-two projection requires float32 or float64 features.")
    if features.ndim < 1 or not 4 <= len(sections) <= 16 or len(sections) % 2:
        raise ValueError("Expected two to eight rank-two sections.")
    width = features.shape[-1]
    end = 0
    tests = []
    for start, stop in zip(sections[::2], sections[1::2], strict=True):
        if not end <= start < stop <= width or (stop - start) % 9:
            raise ValueError("Invalid rank-two projection section.")
        tests.append(f"if(c>={start} && c<{stop}) offset=(c-{start})%9;")
        end = stop
    features = features.contiguous()
    return pointwise(
        torch.empty_like(features),
        (features,),
        f"""
        I c=i%{width}; int offset=-1;
        {" else ".join(tests)}
        T value=p0[i];
        if(offset>=0) {{
            I row=i-offset; int a=offset/3,b=offset%3;
            value=(value+p0[row+b*3+a])*T(0.5);
            if(a==b) value-=(p0[row]+p0[row+4]+p0[row+8])/T(3);
        }}
        out[i]=value;
    """,
    )


@project_rank_two_layout.register_fake
def layout_fake(features, sections):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


def layout_context(ctx, inputs, output):
    ctx.sections = inputs[1]


def layout_backward(ctx, gradient):
    return project_rank_two_layout(gradient, ctx.sections), None


project_rank_two_layout.register_autograd(layout_backward, setup_context=layout_context)


@torch.library.custom_op("eqx::stf_mixed_layout", mutates_args=(), device_types="cuda")
def project_mixed_layout(
    features: torch.Tensor, matrix: torch.Tensor, sections: list[int]
) -> torch.Tensor:
    """Apply rank-two STF and a live rank-three map to a flat component layout."""
    validate_uniform((features, matrix))
    if matrix.shape != (27, 27) or matrix.device != features.device:
        raise ValueError("rank-three map must be a 27 x 27 matrix on the input device")
    if (
        not features.ndim
        or not 0 < len(sections) <= 24
        or len(sections) % 3
        or set(sections[2::3]) != {2, 3}
    ):
        raise ValueError("expected at most eight mixed rank-two/rank-three sections")
    width, end, select = features.size(-1), 0, []
    for start, stop, degree in zip(
        sections[::3], sections[1::3], sections[2::3], strict=True
    ):
        if (
            degree not in (2, 3)
            or not end <= start < stop <= width
            or (stop - start) % (3**degree)
        ):
            raise ValueError("invalid Cartesian projection section")
        select.append(
            f"if(c>={start} && c<{stop}) {{rank={degree}; off=(c-{start})%{3**degree};}}"
        )
        end = stop
    x = features.contiguous()
    return pointwise(
        torch.empty_like(x),
        (x, matrix.contiguous()),
        f"""
        I c=i%{width}; int rank=0, off=0;
        {" else ".join(select)}
        const T* row=p0+(i-off); T value=p0[i];
        if(rank==2) {{
            int a=off/3, b=off%3;
            value=(value+row[3*b+a])*T(0.5);
            if(a==b) value-=(row[0]+row[4]+row[8])/T(3);
        }} else if(rank==3) {{
            value=0;
            #pragma unroll
            for(int j=0;j<27;++j) value+=p1[27*off+j]*row[j];
        }}
        out[i]=value;
        """,
    )


@project_mixed_layout.register_fake
def mixed_fake(features, matrix, sections):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


def mixed_context(ctx, inputs, output):
    features, matrix, ctx.sections = inputs
    ctx.save_for_backward(features if ctx.needs_input_grad[1] else None, matrix)


def mixed_backward(ctx, gradient):
    features, matrix = ctx.saved_tensors
    sections = ctx.sections
    dx = (
        project_mixed_layout(gradient, matrix.T, sections)
        if ctx.needs_input_grad[0]
        else None
    )
    dm = None
    if features is not None:
        for start, stop, degree in zip(
            sections[::3], sections[1::3], sections[2::3], strict=True
        ):
            if degree == 3:
                g = gradient[..., start:stop].reshape(-1, 27)
                x = features[..., start:stop].reshape(-1, 27)
                term = g.T @ x
                dm = term if dm is None else dm + term
    return dx, dm, None


project_mixed_layout.register_autograd(mixed_backward, setup_context=mixed_context)


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
