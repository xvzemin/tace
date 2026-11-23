"""Tiled multilinear products and their transposes."""

import torch

from .....utils.autograd import forward_ad_active
from .... import config


def evaluate(role, sizes, operands, tables, functional=False):
    dst, idx, weights = tables[3 * role : 3 * role + 3]
    x = operands[0]
    rows = max(
        1,
        config.ACE_POLYNOMIAL_WORKSPACE
        // max(1, weights.numel() * x.shape[2] * x.element_size()),
    )
    # Each tile owns its gathered operands.
    outputs = []
    for tile in zip(*(v.split(rows) for v in operands)):
        terms = tile[0].index_select(1, idx[:, 0])
        for k, v in enumerate(tile[1:], 1):
            other = v.index_select(1, idx[:, k])
            terms = terms * other if functional else terms.mul_(other)
        coeff = weights.reshape(1, -1, 1)
        terms = terms * coeff if functional else terms.mul_(coeff)
        out = x.new_zeros((tile[0].shape[0], sizes[role], x.shape[2]))
        outputs.append(
            out.index_add(1, dst, terms)
            if functional
            else out.index_add_(1, dst, terms)
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


class MultilinearMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, role, sizes, *values):
        ctx.role, ctx.sizes = role, sizes
        ctx.save_for_backward(*values)
        n = len(sizes) - 1
        return evaluate(role, sizes, values[:n], values[n:])

    @staticmethod
    def backward(ctx, grad):
        n = len(ctx.sizes) - 1
        saved = ctx.saved_tensors
        inputs = list(saved[:n])
        inputs.insert(ctx.role, grad)
        tables = saved[n:]
        grads = [
            MultilinearMap.apply(k, ctx.sizes, *inputs[:k], *inputs[k + 1 :], *tables)
            if required
            else None
            for k, required in zip(
                (i for i in range(n + 1) if i != ctx.role),
                ctx.needs_input_grad[2 : 2 + n],
            )
        ]
        return None, None, *grads, *([None] * len(tables))


def contract(inputs, tables, sizes):
    role = len(inputs)
    if config.TORCH_REFERENCE or forward_ad_active():
        return evaluate(role, sizes, inputs, tables, functional=True)
    return MultilinearMap.apply(role, sizes, *inputs, *tables)
