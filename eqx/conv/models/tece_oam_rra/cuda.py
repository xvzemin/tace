"""Packed feature views and shared adjoints for TECE-OAM-RRA."""

import torch

from ....kernels.channel_product import channel_product


class LocalSplit(torch.autograd.Function):
    """Return packed views with a single shared gradient buffer."""

    @staticmethod
    def forward(ctx, features, ranges, metadata):
        ctx.kernel_metadata = metadata
        ctx.set_materialize_grads(False)
        return tuple(features.narrow(1, start, size) for start, size in ranges)

    @staticmethod
    def backward(ctx, *gradients):
        reference = next((g for g in gradients if g is not None), None)
        if reference is None:
            return None, None, None
        empty = reference.new_empty((reference.shape[0], 0))
        operands = [empty] + [g if g is not None else empty for g in gradients]
        mapping = tuple(range(len(operands)))
        program = tuple(
            (mapping, 0, 0, 1 << (i + 1))
            for i, g in enumerate(gradients)
            if g is not None
        )
        return (
            channel_product(ctx.kernel_metadata, repr(program), operands)[0],
            None,
            None,
        )
