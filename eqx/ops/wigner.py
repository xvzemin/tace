"""Packed Wigner matrices with quaternion and recursive CUDA kernels."""

import torch

from ..utils.metadata import parse_metadata
from .quaternion import quaternion_wigner


@torch.library.custom_op("eqx::alignment_cuda", mutates_args=(), device_types="cuda")
def alignment_cuda(key: str, values: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute frame geometry expressions or their recursive transposes."""
    from ..backends import get_kernel

    return get_kernel("alignment_cuda", "cuda")(key, values)


@alignment_cuda.register_fake
def alignment_fake(key, values):
    from .scalar import alignment_program

    dtype = "float" if values[0].dtype == torch.float32 else "double"
    _, outputs, _ = alignment_program(parse_metadata(key), dtype)
    return [values[0].new_empty((values[0].size(0), len(group))) for group in outputs]


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


alignment_cuda.register_autograd(
    alignment_backward, setup_context=alignment_setup_context
)


@torch.library.custom_op("eqx::rotation", mutates_args=(), device_types="cuda")
def rotation(cg: torch.Tensor, output: int, values: list[torch.Tensor]) -> torch.Tensor:
    """Evaluate a degree contraction or any of its multilinear transposes."""
    from ..backends import get_kernel

    return get_kernel("rotation", "cuda")(cg, output, values)


@rotation.register_fake
def rotation_fake(cg, output, values):
    return torch.empty_like(values[output], memory_format=torch.contiguous_format)


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


rotation.register_autograd(rotation_backward, setup_context=rotation_setup_context)


@torch.library.custom_op("eqx::packed_wigner", mutates_args=(), device_types="cuda")
def packed_wigner(
    vectors: torch.Tensor, degree: int, coefficients: list[torch.Tensor]
) -> torch.Tensor:
    """Build cached degree matrices directly in one packed allocation."""
    from ..backends import get_kernel

    return get_kernel("packed_wigner", "cuda")(vectors, degree, coefficients)


@packed_wigner.register_fake
def packed_wigner_fake(vectors, degree, coefficients):
    width = sum((2 * l + 1) ** 2 for l in range(degree + 1))
    return vectors.new_empty(vectors.size(0), width)


def wigner_D(frame, vectors, *, backend="cuda", method=None):
    """Return packed Wigner matrices with fused CUDA derivatives.

    Parameters
    ----------
    frame : eqx.o2.WignerD
        Angular cutoff and construction method.
    vectors : torch.Tensor
        Frame directions with shape ``(edges, 3)``.
    backend : {"cuda", "torch"}, optional
        Execution backend. CUDA is the default on GPU. PyTorch uses recursive
        degree contractions, including when falling back to CPU.
    method : {"auto", "quaternion", "recursive"}, optional
        Overrides the frame's method. On CUDA, ``"auto"`` uses direct
        quaternion polynomials. ``"recursive"`` retains the sparse degree
        contractions. Both CUDA methods support higher derivatives.

    Returns
    -------
    torch.Tensor
        Degree blocks flattened and concatenated along the last dimension.
        CPU and unsupported dtypes use the frame's PyTorch implementation.
    """
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    method = getattr(frame, "method", "auto") if method is None else method
    if method not in ("auto", "quaternion", "recursive"):
        raise ValueError("method must be auto, quaternion or recursive.")
    if method == "quaternion" and backend == "torch":
        raise ValueError("The quaternion method requires the CUDA backend.")
    if method == "quaternion" or (
        method == "auto"
        and backend == "cuda"
        and vectors.is_cuda
        and vectors.dtype in (torch.float32, torch.float64)
    ):
        return quaternion_wigner(vectors, frame.lmax)
    if (
        backend == "torch"
        or not vectors.is_cuda
        or vectors.dtype not in (torch.float32, torch.float64)
    ):
        return frame.forward_packed(vectors, method="recursive")
    if not vectors.requires_grad or not torch.is_grad_enabled():
        return packed_wigner(
            vectors.detach(),
            frame.lmax,
            [getattr(frame, f"cg_{l}") for l in range(2, frame.lmax + 1)],
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
                getattr(frame, f"cg_{degree}"),
                2,
                [aligned, matrices[-1], placeholder],
            )
        )
    return torch.cat(matrices, dim=1)
