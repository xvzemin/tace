"""CUDA evaluation of packed Wigner matrices."""

from weakref import ref

import torch

from ...ops.wigner import alignment_fake, packed_wigner_fake
from ...utils.metadata import parse_metadata

_PLANS = {}


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


def rotate(cg, output, values, result, rotation_plan):
    """Evaluate a recursive Wigner contraction using CUDA, including transposes."""
    from .codegen import rotation_source
    from .runtime import kernels

    inputs = [value for i, value in enumerate(values) if i != output]
    indices, coefficients = rotation_plan(cg)[output]
    widths = (9, cg.size(1) ** 2, cg.size(2) ** 2)
    code = rotation_source(
        widths,
        coefficients.size(1),
        "float" if cg.dtype == torch.float32 else "double",
        output,
    )
    kernel = kernels([code], result.device)[code]
    arguments = [value.data_ptr() for value in (*inputs, result, indices, coefficients)]
    arguments += [
        result.size(0),
        *inputs[0].stride(),
        *inputs[1].stride(),
        *result.stride(),
    ]
    kernel.launch(
        arguments,
        (result.numel() + 127) // 128,
        1,
        128,
        torch.cuda.current_stream(result.device).cuda_stream,
    )


def alignment_cuda(key: str, values: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute frame geometry expressions or their recursive transposes."""
    from .codegen import alignment_source
    from .runtime import kernels

    results = alignment_fake(key, values)
    if values[0].size(0):
        dtype = "float" if values[0].dtype == torch.float32 else "double"
        code = alignment_source(parse_metadata(key), dtype)
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


def rotation(cg: torch.Tensor, output: int, values: list[torch.Tensor]) -> torch.Tensor:
    """Evaluate a degree contraction or any of its multilinear transposes."""
    result = torch.empty_like(values[output], memory_format=torch.contiguous_format)
    if result.numel():
        rotate(cg, output, values, result, rotation_plan)
    return result


def packed_wigner(
    vectors: torch.Tensor, degree: int, coefficients: list[torch.Tensor]
) -> torch.Tensor:
    """Build cached degree matrices directly in one packed allocation."""
    result = packed_wigner_fake(vectors, degree, coefficients)
    if not vectors.size(0):
        return result
    result[:, :1].fill_(1)
    if degree:
        aligned = alignment_cuda("None", [vectors])[0]
        result[:, 1:10].copy_(aligned)
        previous, start = aligned, 10
        for cg in coefficients:
            stop = start + cg.size(2) ** 2
            current = result[:, start:stop]
            rotate(cg, 2, (aligned, previous, current), current, rotation_plan)
            previous, start = current, stop
    return result
