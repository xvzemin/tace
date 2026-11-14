"""Live IEEE matrix products with recursive derivatives."""

import torch

from equivariantx.ops.common.cuda.matmul import matmul


@torch.library.custom_op("eqx_oam_xl::precise_mm", mutates_args=())
def precise_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply live matrices with FP32-quality products and FP32 accumulation."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected two compatible matrices")
    if a.dtype == torch.float64 or a.device.type == "cpu":
        return a @ b
    if a.dtype != torch.float32 or b.dtype != a.dtype:
        raise ValueError("Expected FP32 operands")
    return matmul(a, b)


@precise_mm.register_fake
def fake_precise_mm(a, b):
    return a.new_empty((a.shape[0], b.shape[1]))


def precise_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def precise_backward(ctx, gradient):
    a, b = ctx.saved_tensors
    return (
        precise_mm(gradient, b.T) if ctx.needs_input_grad[0] else None,
        precise_mm(a.T, gradient) if ctx.needs_input_grad[1] else None,
    )


precise_mm.register_autograd(precise_backward, setup_context=precise_context)
