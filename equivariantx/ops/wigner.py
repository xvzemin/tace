"""Packed Wigner matrices with quaternion and recursive CUDA kernels."""

import torch

from ..utils.metadata import parse_metadata
from ..utils.outputs import (
    alignment_outputs,
    packed_wigner_output,
    wigner_rotation_output,
)
from . import config
from .precision import validate_module_precision, validate_uniform
from .quaternion import quaternion_wigner


@torch.library.custom_op("eqx::alignment_cuda", mutates_args=(), device_types="cuda")
def alignment_native(key: str, values: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute frame geometry expressions or their recursive transposes."""
    from . import get_kernel

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
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.wigner import alignment

        return alignment(key, values)
    return torch.ops.eqx.alignment_cuda(key, values)


@torch.library.custom_op("eqx::rotation", mutates_args=(), device_types="cuda")
def rotation_native(
    cg: torch.Tensor, output: int, values: list[torch.Tensor]
) -> torch.Tensor:
    """Evaluate a degree contraction or any of its multilinear transposes."""
    from . import get_kernel

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
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.wigner import rotation as evaluate

        return evaluate(cg, output, values)
    return torch.ops.eqx.rotation(cg, output, values)


@torch.library.custom_op("eqx::packed_wigner", mutates_args=(), device_types="cuda")
def packed_wigner_native(
    vectors: torch.Tensor, degree: int, coefficients: list[torch.Tensor]
) -> torch.Tensor:
    """Build cached degree matrices directly in one packed allocation."""
    from . import get_kernel

    return get_kernel("packed_wigner", "cuda")(vectors, degree, coefficients)


packed_wigner_native.register_fake(packed_wigner_output)


@torch.compiler.allow_in_graph
def packed_wigner(vectors, degree, coefficients):
    """Keep inference-only packing from suppressing a forward-mode tangent."""
    validate_uniform((vectors, coefficients))
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.wigner import packed_wigner as evaluate

        return evaluate(vectors, degree, coefficients)
    return torch.ops.eqx.packed_wigner(vectors, degree, coefficients)


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
    method : {"auto", "quaternion", "recursive"}, optional
        Overrides the frame's method. On CUDA, ``"auto"`` uses direct
        quaternion polynomials. ``"recursive"`` retains the sparse degree
        contractions. Both CUDA methods support higher derivatives.

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
    if method not in ("auto", "quaternion", "recursive"):
        raise ValueError("method must be auto, quaternion or recursive.")
    if backend == "torch" and config.TORCH_REFERENCE:
        method = "recursive"
    if method == "quaternion" or (
        method == "auto"
        and backend == "cuda"
        and vectors.is_cuda
        and vectors.dtype in (torch.float32, torch.float64)
    ):
        return quaternion_wigner(vectors, frame.lmax, backend=backend)
    if (
        backend == "torch"
        or not vectors.is_cuda
        or vectors.dtype not in (torch.float32, torch.float64)
    ):
        from ..o2.rotation_matrix import rotation_matrix_to_y_axis

        return torch.cat(
            [
                matrix.flatten(1)
                for matrix in frame.from_matrix(rotation_matrix_to_y_axis(vectors))
            ],
            dim=1,
        )
    if not vectors.requires_grad or not torch.is_grad_enabled():
        return packed_wigner(
            vectors,
            frame.lmax,
            [getattr(frame, f"cg_{l}").to(vectors) for l in range(2, frame.lmax + 1)],
        )
    aligned = alignment_cuda("None", [vectors])[0]
    matrices = [aligned.new_ones((vectors.size(0), 1))]
    if frame.lmax:
        matrices.append(aligned)
    for degree in range(2, frame.lmax + 1):
        placeholder = aligned.new_empty(1).expand(
            vectors.size(0), (2 * degree + 1) ** 2
        )
        matrices.append(
            rotation(
                getattr(frame, f"cg_{degree}").to(aligned),
                2,
                [aligned, matrices[-1], placeholder],
            )
        )
    return torch.cat(matrices, dim=1)
