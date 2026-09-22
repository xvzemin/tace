################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Sparse recursive Wigner contractions for streaming convolutions."""

from weakref import WeakKeyDictionary

import torch

from eqx.o2.rotation_matrix import rotation_matrix_to_y_axis

_PLANS = WeakKeyDictionary()
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


class _Rotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, output, *values):
        ctx.plan, ctx.output = plan, output
        ctx.save_for_backward(*values)
        result = torch.empty_like(values[output], memory_format=torch.contiguous_format)
        if result.numel():
            from .triton import rotation_kernel

            inputs = [value for i, value in enumerate(values) if i != output]
            indices, coefficients = plan[output]
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

    @staticmethod
    def backward(ctx, gradient):
        values = list(ctx.saved_tensors)
        values[ctx.output] = gradient
        return (
            None,
            None,
            *(
                _Rotation.apply(ctx.plan, i, *values)
                if i != ctx.output and ctx.needs_input_grad[i + 2]
                else None
                for i in range(3)
            ),
        )


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
    coefficients = [getattr(frame, f"cg_{l}") for l in range(2, frame.lmax + 1)]
    signature = tuple(
        (id(cg), None if cg.is_inference() else cg._version, cg.device, cg.dtype)
        for cg in coefficients
    )
    cached = _PLANS.get(frame)
    if cached is None or cached[0] != signature:
        plans = []
        for cg in coefficients:
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
                        inputs = tuple(
                            index for i, index in enumerate(indices) if i != output
                        )
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
            plans.append(tuple(plan))
        _PLANS[frame] = signature, plans
    else:
        plans = cached[1]
    rotation = (
        _Alignment.apply(None, vectors)[0]
        if vectors.size(0)
        else rotation_matrix_to_y_axis(vectors).flatten(1)
    )
    matrices = [rotation.new_ones((vectors.size(0), 1))]
    if frame.lmax:
        matrices.append(rotation)
    for degree, plan in enumerate(plans, 2):
        placeholder = rotation.new_empty(1).expand(
            vectors.size(0), (2 * degree + 1) ** 2
        )
        matrices.append(_Rotation.apply(plan, 2, rotation, matrices[-1], placeholder))
    return torch.cat(matrices, dim=1)
