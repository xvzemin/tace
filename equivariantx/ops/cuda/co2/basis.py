"""Native planar symmetric-traceless projection and compiler-visible reduction."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.pointwise import evaluate as pointwise
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


@torch.library.custom_op("eqx::planar_project", mutates_args=(), device_types="cuda")
def project_native(features: torch.Tensor, rank: int) -> torch.Tensor:
    """Apply the self-adjoint planar symmetric-traceless projector."""
    validate_precision((features,))
    if not 2 <= rank <= 8 or features.ndim < 1 or features.shape[-1] != 2**rank:
        raise ValueError("Planar projection requires rank 2..8 and matching storage.")
    features = features.contiguous()
    result = torch.empty_like(features)
    width = 2**rank
    body = f"""
        I row=i/{width}; int c=i%{width};
        int phase=__popc(unsigned(c))%4;
        T total=0;
        for(int k=0;k<{width};++k) {{
            int q=__popc(unsigned(k))%4;
            if(q%2==phase%2) total+=p0[row*{width}+k]*T(q<2?1:-1);
        }}
        out[i]=total*T(phase<2?1:-1)*T({2.0 ** (1 - rank):.17g});
    """
    return pointwise(result, (features,), body)


@project_native.register_fake
def fake(features, rank):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


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
