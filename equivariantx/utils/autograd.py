"""Gradient selection for registered tensor operators."""

import torch


def gradient_mask(operands, required, offset=2, trailing=0):
    """Select input adjoints requested by the current autograd task."""
    if not operands[0].is_cuda or torch.compiler.is_compiling():
        return required
    from ..backends.cuda.runtime import runtime

    active = runtime().gradient_mask(offset, len(operands) + trailing)
    return [a and b for a, b in zip(required, active)] if active else required
