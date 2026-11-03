"""Head-wise rotary inner products and their recursive transposes."""

import torch


def rotary_product(query, key, phase, lmax, mmax, heads):
    empty = query.new_empty((0, heads))
    return contract(query, key, phase, empty, lmax, mmax, heads, 3)


@torch.library.custom_op("eqx::rotary_product", mutates_args=(), device_types="cuda")
def contract(
    query: torch.Tensor,
    key: torch.Tensor,
    phase: torch.Tensor,
    score: torch.Tensor,
    lmax: int,
    mmax: int,
    heads: int,
    role: int,
) -> torch.Tensor:
    from ..backends import get_kernel

    return get_kernel("rotary_product", "cuda")(
        query, key, phase, score, lmax, mmax, heads, role
    )


@contract.register_fake
def contract_fake(query, key, phase, score, lmax, mmax, heads, role):
    if role < 2:
        return torch.empty_like(query)
    if role == 2:
        return torch.empty_like(phase)
    return query.new_empty((query.shape[0], heads))


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


contract.register_autograd(backward, setup_context=setup_context)
