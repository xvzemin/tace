"""Gradient selection and reduction utilities."""

import torch


@torch.no_grad()
def clip_gradient_norm(parameters, maximum):
    """Clip parameter gradients in-place to a global L2 norm.

    Parameters
    ----------
    parameters : iterable of torch.nn.Parameter
        Parameters with optional gradients.
    maximum : float
        Nonnegative upper bound on the gradient norm.

    Returns
    -------
    torch.Tensor
        The global norm before clipping.
    """
    if maximum < 0:
        raise ValueError("maximum must be nonnegative")
    groups = {}
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None:
            groups.setdefault((grad.device, grad.dtype), []).append(grad)
    if not groups:
        return torch.tensor(0.0)
    device = next(iter(groups))[0]
    norms = [
        n.to(device)
        for values in groups.values()
        for n in torch._foreach_norm(values, 2)
    ]
    total = torch.linalg.vector_norm(torch.stack(norms))
    if not bool(total.isfinite()):
        # Individual finite FP32 components can overflow during the sum of squares.
        accumulation_device = "cpu" if device.type == "mps" else device
        norms = [
            torch.linalg.vector_norm(
                g.cpu() if g.device.type == "mps" else g, dtype=torch.float64
            ).to(accumulation_device)
            for values in groups.values()
            for g in values
        ]
        total = torch.linalg.vector_norm(torch.stack(norms))
    coefficient = (float(maximum) / (total + 1e-6)).clamp(max=1.0)
    for (device, dtype), values in groups.items():
        torch._foreach_mul_(values, coefficient.to(device=device, dtype=dtype))
    return total


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
