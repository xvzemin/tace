"""Packed Wigner matrices with quaternion and recursive CUDA kernels."""

import torch

from ...utils.autograd import forward_ad_active
from ...utils.metadata import parse_metadata
from ...utils.outputs import (
    alignment_outputs,
    packed_wigner_output,
    wigner_rotation_output,
)
from .. import config
from ..precision import validate_module_precision, validate_uniform
from .compilation import alignment_work, should_inline


def inline(estimate, *args):
    """Expose bounded geometry while preserving functional differentiation."""
    return forward_ad_active() or should_inline(config.WIGNER_INLINE, estimate, *args)


def rotation_work(degree):
    """Count recursive degree contractions, whose angular axes are vectorized."""
    return 16 * (degree + 1)


@torch.library.custom_op("eqx::alignment_cuda", mutates_args=(), device_types="cuda")
def alignment_native(key: str, values: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute frame geometry expressions or their recursive transposes."""
    from .. import get_kernel

    return get_kernel("alignment_cuda", "cuda")(key, values)


alignment_native.register_fake(alignment_outputs)


def alignment_setup_context(ctx, inputs, output):
    key, values = inputs
    ctx.key = parse_metadata(key)
    ctx.output_widths = tuple(value.size(1) for value in output)
    ctx.save_for_backward(*values)


def alignment_backward(ctx, gradients):
    values = ctx.saved_tensors
    active = tuple(i for i, need in enumerate(ctx.needs_input_grad[1]) if need)
    results = [None] * len(values)
    if active:
        cotangents = [
            grad
            if grad is not None
            else values[0].new_zeros((values[0].size(0), width))
            for grad, width in zip(gradients, ctx.output_widths)
        ]
        transposed = alignment_cuda(repr((ctx.key, active)), [*values, *cotangents])
        for i, value in zip(active, transposed):
            results[i] = value
    return None, results


alignment_native.register_autograd(
    alignment_backward, setup_context=alignment_setup_context
)


@torch.compiler.allow_in_graph
def alignment_cuda(key, values):
    """Retain native geometry while exposing functional transforms to Torch."""
    validate_uniform(values)
    if inline(alignment_work, key):
        from .torch.wigner import alignment

        return alignment(key, values)
    return torch.ops.eqx.alignment_cuda(key, values)


@torch.library.custom_op("eqx::rotation", mutates_args=(), device_types="cuda")
def rotation_native(
    cg: torch.Tensor, output: int, values: list[torch.Tensor]
) -> torch.Tensor:
    """Evaluate a degree contraction or any of its multilinear transposes."""
    from .. import get_kernel

    return get_kernel("rotation", "cuda")(cg, output, values)


rotation_native.register_fake(wigner_rotation_output)


def rotation_setup_context(ctx, inputs, output):
    cg, role, values = inputs
    ctx.output = role
    ctx.save_for_backward(cg, *values)


def rotation_backward(ctx, gradient):
    cg, *values = ctx.saved_tensors
    values[ctx.output] = gradient
    return (
        None,
        None,
        [
            rotation(cg, i, values)
            if i != ctx.output and ctx.needs_input_grad[2][i]
            else None
            for i in range(3)
        ],
    )


rotation_native.register_autograd(
    rotation_backward, setup_context=rotation_setup_context
)


@torch.compiler.allow_in_graph
def rotation(cg, output, values):
    """Expose recursive degree contractions inside functional transformations."""
    validate_uniform((cg, values))
    if inline(rotation_work, 1):
        from .torch.wigner import rotation as evaluate

        return evaluate(cg, output, values)
    return torch.ops.eqx.rotation(cg, output, values)


@torch.library.custom_op("eqx::packed_wigner", mutates_args=(), device_types="cuda")
def packed_wigner_native(
    vectors: torch.Tensor, degree: int, coefficients: list[torch.Tensor]
) -> torch.Tensor:
    """Build cached degree matrices directly in one packed allocation."""
    from .. import get_kernel

    return get_kernel("packed_wigner", "cuda")(vectors, degree, coefficients)


packed_wigner_native.register_fake(packed_wigner_output)


@torch.compiler.allow_in_graph
def packed_wigner(vectors, degree, coefficients):
    """Keep inference-only packing from suppressing a forward-mode tangent."""
    validate_uniform((vectors, coefficients))
    if inline(rotation_work, degree):
        from .torch.wigner import packed_wigner as evaluate

        return evaluate(vectors, degree, coefficients)
    return torch.ops.eqx.packed_wigner(vectors, degree, coefficients)


@torch.library.custom_op("eqx::generator_wigner", mutates_args=(), device_types="cuda")
def generator_wigner(
    degree: int, mode: int, first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Evaluate packed rotation blocks, a tangent action or its four contractions."""
    from .cuda.wigner_generator import evaluate

    return evaluate(degree, mode, first, second)


@generator_wigner.register_fake
def generator_output(degree, mode, first, second):
    width = (degree + 1) * (2 * degree + 1) * (2 * degree + 3) // 3
    return first.new_empty((first.shape[0], 4 if mode == 2 else width))


def generator_context(ctx, inputs, output):
    degree, mode, first, second = inputs
    ctx.degree, ctx.mode = degree, mode
    ctx.save_for_backward(first, output if mode == 0 else second)


def generator_backward(ctx, gradient):
    degree = ctx.degree
    first, second = ctx.saved_tensors
    op = torch.ops.eqx.generator_wigner.default

    def conjugate(q):
        return torch.cat((q[:, :1], -q[:, 1:]), -1)

    if ctx.mode == 0:
        # dD_l(q) = [l Re(u) I + Im(u).J_l] D_l(q), u=2 dq q*/|q|².
        rate = op(degree, 2, second, gradient)
        a, u = rate[:, :1], rate[:, 1:]
        b, v = first[:, :1], first[:, 1:]
        dq = torch.cat(
            (a * b - (u * v).sum(-1, True), a * v + b * u + torch.linalg.cross(u, v)),
            -1,
        )
        return None, None, dq * (2 / first.square().sum(-1, True)), None
    if ctx.mode == 1:
        return (
            None,
            None,
            op(degree, 1, gradient, conjugate(second))
            if ctx.needs_input_grad[2]
            else None,
            op(degree, 2, first, gradient) if ctx.needs_input_grad[3] else None,
        )
    return (
        None,
        None,
        op(degree, 1, second, conjugate(gradient)) if ctx.needs_input_grad[2] else None,
        op(degree, 1, first, gradient) if ctx.needs_input_grad[3] else None,
    )


generator_wigner.register_autograd(generator_backward, setup_context=generator_context)


def wigner_D(frame, vectors, *, backend="torch", method=None):
    """Return packed Wigner matrices with fused CUDA derivatives.

    Parameters
    ----------
    frame : equivariantx.o2.WignerD
        Angular cutoff and construction method.
    vectors : torch.Tensor
        Frame directions with shape ``(edges, 3)``.
    backend : {"cuda", "torch"}, optional
        Execution provider. Defaults to Torch on either device.
    method : {"auto", "quaternion", "recursive", "generator"}, optional
        Overrides the frame's method. On CUDA, ``"auto"`` uses direct
        quaternion polynomials. ``"recursive"`` retains the sparse degree
        contractions. ``"generator"`` differentiates packed degree matrices
        with the rotation generators on CUDA. All methods support higher
        derivatives.

    Returns
    -------
    torch.Tensor
        Degree blocks flattened and concatenated along the last dimension.
        CPU inputs use Torch on either backend. Only FP32/FP64 are supported.
    """
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    validate_module_precision(frame, (vectors,))
    method = getattr(frame, "method", "auto") if method is None else method
    if method == "generator":
        if backend == "cuda" and vectors.is_cuda and not forward_ad_active():
            q = alignment_cuda(repr("quaternion"), [vectors])[0]
            return generator_wigner(frame.lmax, 0, q, q.new_empty(0))
        method = "recursive"
    if method not in ("auto", "quaternion", "recursive"):
        raise ValueError("method must be auto, quaternion, recursive or generator.")
    if backend == "torch" and config.TORCH_REFERENCE:
        method = "recursive"
    if backend == "cuda" and vectors.is_cuda:
        from ..cuda.o2 import wigner as cuda_kernel

        return cuda_kernel.packed(frame, vectors, method)
    from ..torch.o2 import wigner as torch_kernel

    return torch_kernel.packed(frame, vectors, method)
