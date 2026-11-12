"""Head-wise rotary inner products and their recursive transposes."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ..utils.outputs import rotary_output
from . import config
from .precision import validate_uniform


def rotary_product(query, key, phase, lmax, mmax, heads):
    empty = query.new_empty((0, heads))
    return contract(query, key, phase, empty, lmax, mmax, heads, 3)


@torch.library.custom_op("eqx::rotary_product", mutates_args=(), device_types="cuda")
def contract_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    phase: torch.Tensor,
    score: torch.Tensor,
    lmax: int,
    mmax: int,
    heads: int,
    role: int,
) -> torch.Tensor:
    from . import get_kernel

    return get_kernel("rotary_product", "cuda")(
        query, key, phase, score, lmax, mmax, heads, role
    )


contract_cuda.register_fake(rotary_output)


def setup_context(ctx, inputs, output):
    query, key, phase, score, ctx.lmax, ctx.mmax, ctx.heads, ctx.role = inputs
    ctx.save_for_backward(query, key, phase, score)


def backward(ctx, grad):
    values = list(ctx.saved_tensors)
    values[ctx.role] = grad
    gradients = [
        contract(*values, ctx.lmax, ctx.mmax, ctx.heads, role)
        if role != ctx.role and ctx.needs_input_grad[role]
        else None
        for role in range(4)
    ]
    return (*gradients, None, None, None, None)


contract_cuda.register_autograd(backward, setup_context=setup_context)


@torch.compiler.allow_in_graph
def contract(query, key, phase, score, lmax, mmax, heads, role):
    """Expose head-wise rotary products inside functional transformations."""
    validate_uniform((query, key, phase, score))
    expanded = (
        role == 3
        and config.ROTARY_INLINE
        and not config.TORCH_RECOMPUTE
        and lmax <= 4
        and mmax <= 2
        and query.shape[-1] <= 128
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )
    if expanded or torch._C._are_functorch_transforms_active():
        from .kernels.torch.frame import rotary_product

        return rotary_product(query, key, phase, score, lmax, mmax, heads, role)
    return torch.ops.eqx.rotary_product(
        query, key, phase, score, lmax, mmax, heads, role
    )
