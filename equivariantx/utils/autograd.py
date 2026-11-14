"""Gradient selection for registered tensor operators."""

import torch


def forward_ad_active():
    """Detect functional transforms and ordinary forward-AD dual contexts."""
    return (
        torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    )


def nested_forward_ad():
    """Detect nested JVPs whose outer tangent PyTorch disables in Function.jvp."""
    from torch._C._functorch import TransformType, get_interpreter_stack

    return (
        sum(
            transform.key() == TransformType.Jvp
            for transform in (get_interpreter_stack() or ())
        )
        > 1
    )


def torch_gradient_mask(ctx, required):
    """Skip unrequested adjoints without loading the optional CUDA runtime.

    The engine query cannot inspect captured leaf gradients on every PyTorch
    version. Those inputs conservatively retain their derivative program.
    Functional transforms and compiler tracing use the static gradient mask.
    """
    if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
        return required
    query = getattr(torch._C, "_will_engine_execute_node", None)
    if query is None:
        return required
    edges = ctx.next_functions[-len(required) :] if required else ()
    if len(edges) != len(required):
        return required
    selected = []
    for needed, (node, _) in zip(required, edges):
        if needed and node is not None:
            try:
                needed = query(node)
            except RuntimeError:
                pass
        selected.append(needed)
    return tuple(selected)


def gradient_mask(operands, required, offset=2, trailing=0):
    """Select input adjoints requested by the current autograd task."""
    if not operands[0].is_cuda or torch.compiler.is_compiling():
        return required
    from ..ops.common.cuda.runtime import runtime

    active = runtime().gradient_mask(offset, len(operands) + trailing)
    return [a and b for a, b in zip(required, active)] if active else required
