################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Sparse recursive Wigner contractions for streaming convolutions."""

from weakref import ref

import torch
from torch._subclasses.fake_tensor import is_fake

from eqx.o2.rotation_matrix import rotation_matrix_to_y_axis

_PLANS = {}
_ALIGNMENTS = {}


def _alignment(vectors):
    return (rotation_matrix_to_y_axis(vectors).flatten(1),)


class _Alignment(torch.autograd.Function):
    """Compile each required transpose, including higher-order transposes."""

    @staticmethod
    def forward(ctx, key, *values):
        if key not in _ALIGNMENTS:
            if key is None:
                function = _alignment
            else:
                parent, size, active = key
                original = _ALIGNMENTS[parent][0]

                def function(*inputs):
                    _, pullback = torch.func.vjp(original, *inputs[:size])
                    gradients = pullback(tuple(inputs[size:]))
                    return tuple(gradients[i] for i in active)

            _ALIGNMENTS[key] = function, {}
        ctx.key = key
        ctx.save_for_backward(*values)
        function, compiled = _ALIGNMENTS[key]
        signature = tuple(
            (value.device, value.dtype, value.size(0) == 1) for value in values
        )
        if signature not in compiled:
            from torch.fx.experimental.proxy_tensor import make_fx

            # Each transpose has its own tensor graph and compilation cache.
            # Tracing symbolic batch sizes avoids specialization on edge count.
            graph = make_fx(function, tracing_mode="symbolic")(*values)
            compiled[signature] = torch.compile(graph, fullgraph=True, dynamic=True)
        return compiled[signature](*values)

    @staticmethod
    def backward(ctx, *cotangents):
        values = ctx.saved_tensors
        active = tuple(i for i, need in enumerate(ctx.needs_input_grad[1:]) if need)
        results = _Alignment.apply((ctx.key, len(values), active), *values, *cotangents)
        gradients = [None] * len(values)
        for i, value in zip(active, results):
            gradients[i] = value
        return None, *gradients


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
                float(coefficients_cpu[a, b, m] * coefficients_cpu[c, d, n])
                * current
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
def rotation(
    cg: torch.Tensor, output: int, values: list[torch.Tensor]
) -> torch.Tensor:
    """Evaluate a degree contraction or any of its multilinear transposes."""
    result = torch.empty_like(values[output], memory_format=torch.contiguous_format)
    if result.numel():
        from .triton import rotation_kernel

        inputs = [value for i, value in enumerate(values) if i != output]
        indices, coefficients = rotation_plan(cg)[output]
        rotation_kernel[((result.size(0) + 3) // 4, result.size(1))](
            *inputs,
            result,
            indices,
            coefficients,
            result.size(0),
            *inputs[0].stride(),
            *inputs[1].stride(),
            result.size(1),
            coefficients.size(1),
            4,
            num_warps=4,
        )
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


def wigner_D(frame, vectors):
    """Return packed Wigner matrices with recursively fused CUDA derivatives.

    Parameters
    ----------
    frame : eqx.o2.WignerD
        Degree cutoff and Clebsch--Gordan buffers defining the rotations.
    vectors : torch.Tensor
        Frame directions with shape ``(edges, 3)``.

    Returns
    -------
    torch.Tensor
        Degree blocks flattened and concatenated along the last dimension.
        CPU and unsupported dtypes use the frame's PyTorch implementation.
    """
    if not vectors.is_cuda or vectors.dtype not in (torch.float32, torch.float64):
        return frame.forward_packed(vectors)
    # Let an enclosing compiler fuse the alignment and its derivatives instead
    # of starting a nested trace. Eager execution retains its compiled cache.
    if torch.compiler.is_compiling() or is_fake(vectors) or not vectors.size(0):
        aligned = rotation_matrix_to_y_axis(vectors).flatten(1)
    else:
        aligned = _Alignment.apply(None, vectors)[0]
    matrices = [aligned.new_ones((vectors.size(0), 1))]
    if frame.lmax:
        matrices.append(aligned)
    for degree in range(2, frame.lmax + 1):
        placeholder = aligned.new_empty(1).expand(
            vectors.size(0), (2 * degree + 1) ** 2
        )
        matrices.append(
            rotation(
                getattr(frame, f"cg_{degree}"), 2, [aligned, matrices[-1], placeholder]
            )
        )
    return torch.cat(matrices, dim=1)
