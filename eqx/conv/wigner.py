################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Sparse recursive Wigner contractions for streaming convolutions."""

from ast import literal_eval
from functools import lru_cache
from weakref import ref

import torch

_PLANS = {}

parse_alignment = lru_cache(maxsize=128)(literal_eval)


@torch.library.custom_op("eqx::alignment_cuda", mutates_args=(), device_types="cuda")
def alignment_cuda(key: str, values: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute the quaternion alignment or a recursively generated transpose."""
    from .codegen import alignment_source
    from .cuda import kernels

    results = alignment_fake(key, values)
    if values[0].size(0):
        dtype = "float" if values[0].dtype == torch.float32 else "double"
        code = alignment_source(parse_alignment(key), dtype)
        kernel = kernels([code], values[0].device)[code]
        contiguous = [value.contiguous() for value in values]
        args = [value.data_ptr() for value in (*contiguous, *results)] + [
            values[0].size(0)
        ]
        kernel.launch(
            args,
            (values[0].size(0) + 127) // 128,
            1,
            128,
            torch.cuda.current_stream(values[0].device).cuda_stream,
        )
    return results


@alignment_cuda.register_fake
def alignment_fake(key, values):
    from .codegen import alignment_program

    dtype = "float" if values[0].dtype == torch.float32 else "double"
    _, outputs, _ = alignment_program(parse_alignment(key), dtype)
    return [values[0].new_empty((values[0].size(0), len(group))) for group in outputs]


def alignment_setup_context(ctx, inputs, output):
    key, values = inputs
    ctx.key = parse_alignment(key)
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


def rotation_plan(cg):
    """Build sparse degree contractions only from real, runtime CG buffers."""
    key = id(cg)
    version = None if cg.is_inference() else cg._version
    cached = _PLANS.get(key)
    if cached is not None and cached[0]() is cg and cached[1] == version:
        return cached[2]
    coefficients_cpu = cg.detach().to(device="cpu", dtype=torch.float64)
    entries = coefficients_cpu.nonzero().tolist()
    previous, current = cg.size(1), cg.size(2)
    widths = (9, previous**2, current**2)
    paths = [[[] for _ in range(width)] for width in widths]
    for a, b, m in entries:
        for c, d, n in entries:
            indices = (3 * a + c, previous * b + d, current * m + n)
            value = (
                float(coefficients_cpu[a, b, m] * coefficients_cpu[c, d, n]) * current
            )
            for output in range(3):
                inputs = tuple(index for i, index in enumerate(indices) if i != output)
                paths[output][indices[output]].append((*inputs, value))
    plan = []
    for rows in paths:
        count = 1 << (max(map(len, rows)) - 1).bit_length()
        indices = torch.zeros(len(rows), count, 2, dtype=torch.int32)
        values = torch.zeros(len(rows), count, dtype=cg.dtype)
        for row, entries in enumerate(rows):
            for column, (a, b, value) in enumerate(entries):
                indices[row, column, 0] = a
                indices[row, column, 1] = b
                values[row, column] = value
        plan.append((indices.to(cg.device), values.to(cg.device)))
    _PLANS[key] = ref(cg, lambda _: _PLANS.pop(key, None)), version, tuple(plan)
    return tuple(plan)


@torch.library.custom_op("eqx::rotation", mutates_args=(), device_types="cuda")
def rotation(cg: torch.Tensor, output: int, values: list[torch.Tensor]) -> torch.Tensor:
    """Evaluate a degree contraction or any of its multilinear transposes."""
    result = torch.empty_like(values[output], memory_format=torch.contiguous_format)
    if result.numel():
        from .cuda import rotate

        rotate(cg, output, values, result, rotation_plan)
    return result


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


def wigner_D(frame, vectors, *, backend="cuda"):
    """Return packed Wigner matrices with recursively fused CUDA derivatives.

    Parameters
    ----------
    frame : eqx.o2.WignerD
        Degree cutoff and Clebsch--Gordan buffers defining the rotations.
    vectors : torch.Tensor
        Frame directions with shape ``(edges, 3)``.
    backend : {"cuda", "torch"}, optional
        Execution backend. Generated CUDA is the default on GPU. The
        quaternion alignment and recursive degree contractions retain
        differentiable transposes at every order.

    Returns
    -------
    torch.Tensor
        Degree blocks flattened and concatenated along the last dimension.
        CPU and unsupported dtypes use the frame's PyTorch implementation.
    """
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    if (
        backend == "torch"
        or not vectors.is_cuda
        or vectors.dtype not in (torch.float32, torch.float64)
    ):
        return frame.forward_packed(vectors)
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
