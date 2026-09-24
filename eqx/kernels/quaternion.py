"""CUDA quaternion Wigner matrices and their directional derivatives."""

import math
from collections import defaultdict
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def polynomial_coefficients(lmax):
    """Expand the symmetric-power representation in the real harmonic basis.

    Polynomial coefficients are accumulated as integers before applying the
    harmonic normalization. Structural zeros therefore need no threshold or
    numerical fit. The spin-half matrix is
    ``[[w - iy, x + iz], [-x + iz, w + iy]]``.
    """
    pointers, exponents, coefficients = [0], [], []
    for l in range(lmax + 1):
        n = 2 * l

        @lru_cache(maxsize=None)
        def complex_polynomial(row, column):
            result = defaultdict(int)
            for r in range(max(0, row - column), min(n - column, row) + 1):
                a, b, c, d = n - column - r, column - row + r, r, row - r
                wy, xz = [0] * (a + d + 1), [0] * (b + c + 1)
                for i in range(a + 1):
                    for j in range(d + 1):
                        wy[i + j] += (-1) ** i * math.comb(a, i) * math.comb(d, j)
                for i in range(b + 1):
                    for j in range(c + 1):
                        xz[i + j] += (-1) ** (c - j) * math.comb(b, i) * math.comb(c, j)
                scale = math.comb(n - column, r) * math.comb(column, row - r)
                for y, cy in enumerate(wy):
                    for z, cz in enumerate(xz):
                        if cy and cz:
                            result[a + d - y, b + c - z, y, z] += scale * cy * cz
            return result

        # Each entry is (complex index, power of i). The common degree phase
        # cancels between the real-to-complex matrix and its adjoint.
        basis = []
        for m in range(-l, l + 1):
            if m < 0:
                basis.append(((l + m, 3), (l - m, (1 - 2 * m) % 4)))
            elif m > 0:
                basis.append(((l - m, 0), (l + m, (2 * m) % 4)))
            else:
                basis.append(((l, 0),))
        for row in range(n + 1):
            for column in range(n + 1):
                real, imaginary = defaultdict(int), defaultdict(int)
                for i, pi in basis[row]:
                    for j, pj in basis[column]:
                        for powers, value in complex_polynomial(i, j).items():
                            phase = (powers[2] + powers[3] + pj - pi) % 4
                            (real if phase % 2 == 0 else imaginary)[powers] += (
                                value if phase < 2 else -value
                            )
                if any(imaginary.values()):
                    raise RuntimeError("The quaternion polynomial must be real.")
                scale = math.sqrt(math.comb(n, column) / math.comb(n, row))
                scale /= math.sqrt(len(basis[row]) * len(basis[column]))
                for powers, value in sorted(real.items()):
                    if value:
                        exponents.append(powers)
                        coefficients.append(value * scale)
                pointers.append(len(coefficients))
    return (
        torch.tensor(pointers, dtype=torch.int32, device="cpu"),
        torch.tensor(exponents, dtype=torch.int32, device="cpu"),
        torch.tensor(coefficients, dtype=torch.float64, device="cpu"),
    )


@lru_cache(maxsize=64)
def polynomial_tables(lmax, device, dtype):
    """Cache immutable coefficient tables on the execution device."""
    pointers, exponents, coefficients = polynomial_coefficients(lmax)
    return pointers.to(device), exponents.to(device), coefficients.to(device, dtype)


@lru_cache(maxsize=128)
def polynomial_source(lmax, rank, transpose, dtype):
    """Evaluate a polynomial derivative or its vector cotangent in CUDA."""
    from .codegen import HEADER

    width = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    powers = 2 * lmax + 1
    evaluator = []
    if rank == 0 and not transpose:
        # A warp evaluates one factored polynomial across 32 edges, avoiding
        # repeated coefficient gathers and power-table loads.
        pointers, exponents, coefficients = polynomial_coefficients(lmax)
        entries = list(zip(exponents.tolist(), coefficients.tolist()))

        def horner(terms, axis=0):
            if axis == 4:
                return f"T({terms[0][1]:.17g})"
            groups = defaultdict(list)
            for term in terms:
                groups[term[0][axis]].append(term)
            orders = sorted(groups, reverse=True)
            last = orders[0]
            value = horner(groups[last], axis + 1)
            for order in orders[1:]:
                value = f"fma({value}, power[{axis}][{last - order}][lane], {horner(groups[order], axis + 1)})"
                last = order
            if last:
                value = f"({value} * power[{axis}][{last}][lane])"
            return value

        # Keep optimization units bounded without adding kernel launches.
        # A single switch over all degrees becomes expensive to compile at
        # high degree; each device function instead owns one output tile.
        for tile in range((width + 31) // 32):
            evaluator += [
                f"__device__ __noinline__ T evaluate_{tile}(int element, T power[4][{powers}][32], int lane) {{",
                "switch (element) {",
            ]
            for element in range(32 * tile, min(width, 32 * (tile + 1))):
                start, end = pointers[element : element + 2]
                evaluator.append(
                    f"case {element % 32}: return {horner(entries[start:end])};"
                )
            evaluator += ["default: return T(0);", "}", "}"]
        evaluator += [
            f"__device__ T evaluate(int element, T power[4][{powers}][32], int lane) {{",
            "switch (element / 32) {",
        ]
        for tile in range((width + 31) // 32):
            evaluator.append(
                f"case {tile}: return evaluate_{tile}(element % 32, power, lane);"
            )
        evaluator += ["default: return T(0);", "}", "}"]
    arguments = ["const T* q"] + [f"const T* v{i}" for i in range(rank)]
    if transpose:
        arguments.append("const T* g")
    arguments.append("T* out")
    if not evaluator:
        arguments += ["const int* ptr", "const int* exponents", "const T* coefficients"]
    arguments.append("int64_t edges")
    lines = [
        HEADER.replace("SCALAR", dtype),
        *evaluator,
        'extern "C" __global__ void run(' + ", ".join(arguments) + ") {",
        "const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;",
        "const int64_t edge = int64_t(blockIdx.x) * 32 + lane;",
        f"__shared__ T power[4][{powers}][32];",
        "__shared__ T tile[32][33];",
        "T value = 1, base = edge < edges ? q[edge * 4 + warp] : T(0);",
        f"for (int i = 0; i < {powers}; ++i) {{ power[warp][i][lane] = value; value *= base; }}",
        "__syncthreads();",
    ]
    if transpose:
        lines += [
            "T total[4] = {0, 0, 0, 0};",
            f"for (int start = 0; start < {width}; start += 32) {{",
            "for (int i = threadIdx.x; i < 1024; i += 128) {",
            "const int row = i % 32, local_edge = i / 32;",
            "const int64_t e = int64_t(blockIdx.x) * 32 + local_edge;",
            f"tile[row][local_edge] = e < edges && start + row < {width} ? g[e * {width} + start + row] : T(0);",
            "}",
            "__syncthreads();",
        ]
    else:
        lines.append("const int start = blockIdx.y * 32;")
    lines += [
        "for (int row = warp; row < 32; row += 4) {",
        "const int element = start + row;",
        "T sum[4] = {0, 0, 0, 0};" if transpose else "T sum = 0;",
        f"if (element < {width} && edge < edges) {{",
    ]
    if evaluator:
        lines += ["sum = evaluate(element, power, lane);", "}"]
    else:
        lines += [
            "for (int k = ptr[element]; k < ptr[element + 1]; ++k) {",
            f"for (int assignment = 0; assignment < {4**rank}; ++assignment) {{",
            "int p[4] = {exponents[4*k], exponents[4*k+1], exponents[4*k+2], exponents[4*k+3]};",
            "T value = coefficients[k];",
        ]
    for i in range(rank):
        lines += [
            f"const int a{i} = (assignment >> {2 * i}) & 3;",
            f"if (!p[a{i}]) continue;",
            f"value *= T(p[a{i}]--) * v{i}[edge * 4 + a{i}];",
        ]
    if transpose:
        lines += [
            "for (int a = 0; a < 4; ++a) {",
            "if (!p[a]) continue;",
            "T term = value * T(p[a]--);",
            "for (int b = 0; b < 4; ++b) term *= power[b][p[b]][lane];",
            "++p[a]; sum[a] += term;",
            "}",
        ]
    elif not evaluator:
        lines += [
            "for (int b = 0; b < 4; ++b) value *= power[b][p[b]][lane];",
            "sum += value;",
        ]
    if not evaluator:
        lines += ["}", "}", "}"]
    if transpose:
        lines.append(
            "for (int a = 0; a < 4; ++a) total[a] = fma(sum[a], tile[row][lane], total[a]);"
        )
    else:
        lines.append("tile[row][lane] = sum;")
    lines.append("}")
    if transpose:
        lines += [
            "__syncthreads();",
            "}",
            "for (int a = 0; a < 4; ++a) tile[4*a + warp][lane] = total[a];",
            "__syncthreads();",
            "if (warp == 0 && edge < edges) {",
            "for (int a = 0; a < 4; ++a) {",
            "T sum = 0; for (int w = 0; w < 4; ++w) sum += tile[4*a+w][lane];",
            "out[edge * 4 + a] = sum;",
            "}",
            "}",
        ]
    else:
        lines += [
            "__syncthreads();",
            "for (int i = threadIdx.x; i < 1024; i += 128) {",
            "const int row = i % 32, local_edge = i / 32;",
            "const int64_t e = int64_t(blockIdx.x) * 32 + local_edge;",
            f"if (e < edges && start + row < {width}) out[e * {width} + start + row] = tile[row][local_edge];",
            "}",
        ]
    lines.append("}")
    return "\n".join(lines)


@torch.library.custom_op(
    "eqx::quaternion_polynomial", mutates_args=(), device_types="cuda"
)
def quaternion_polynomial(
    lmax: int, transpose: bool, values: list[torch.Tensor]
) -> torch.Tensor:
    """Contract Wigner polynomial derivatives without materializing Jacobians."""
    from .cuda import kernels

    result = quaternion_polynomial_fake(lmax, transpose, values)
    if len(values) - 1 > 2 * lmax:
        return result.zero_()
    if result.numel():
        q = values[0]
        tables = (
            ()
            if not transpose and len(values) == 1
            else polynomial_tables(lmax, q.device, q.dtype)
        )
        code = polynomial_source(
            lmax,
            len(values) - 1 - transpose,
            transpose,
            "float" if q.dtype == torch.float32 else "double",
        )
        kernel = kernels([code], q.device)[code]
        contiguous = [value.contiguous() for value in values]
        args = [value.data_ptr() for value in (*contiguous, result, *tables)] + [
            q.size(0)
        ]
        kernel.launch(
            args,
            (q.size(0) + 31) // 32,
            1 if transpose else (result.size(1) + 31) // 32,
            128,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    return result


@quaternion_polynomial.register_fake
def quaternion_polynomial_fake(lmax, transpose, values):
    width = 4 if transpose else sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    return values[0].new_empty((values[0].size(0), width))


def polynomial_setup_context(ctx, inputs, output):
    ctx.lmax, ctx.transpose, values = inputs
    ctx.save_for_backward(*values)


def polynomial_backward(ctx, gradient):
    q, *values = ctx.saved_tensors
    directions = values[:-1] if ctx.transpose else values
    cotangent = values[-1] if ctx.transpose else gradient
    extra = [gradient] if ctx.transpose else []
    results = []
    for i, need in enumerate(ctx.needs_input_grad[2]):
        if not need:
            results.append(None)
        elif ctx.transpose and i == len(values):
            results.append(
                quaternion_polynomial(ctx.lmax, False, [q, *directions, gradient])
            )
        else:
            selected = [v for j, v in enumerate(directions, 1) if i == 0 or j != i]
            results.append(
                quaternion_polynomial(ctx.lmax, True, [q, *selected, *extra, cotangent])
            )
    return None, None, results


quaternion_polynomial.register_autograd(
    polynomial_backward, setup_context=polynomial_setup_context
)


def quaternion_wigner(vectors, lmax):
    """Return packed Wigner matrices using direct quaternion polynomials.

    Parameters
    ----------
    vectors : torch.Tensor
        Nonzero frame directions of shape ``(edges, 3)`` on CUDA, in float32
        or float64. Their alignment is the same as the recursive method.
    lmax : int
        Maximum angular degree. All orders are retained.

    Returns
    -------
    torch.Tensor
        Degree matrices concatenated as ``(edges, sum((2*l+1)**2))``.
        Both the alignment and polynomial contractions support higher derivatives.
    """
    from .wigner import alignment_cuda

    if not vectors.is_cuda or vectors.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "Quaternion Wigner matrices require CUDA float32 or float64 inputs."
        )
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (edges, 3).")
    if lmax == 0:
        return vectors.new_ones((vectors.size(0), 1))
    if lmax == 1:
        return alignment_cuda(repr("wigner1"), [vectors])[0]
    q = alignment_cuda(repr("quaternion"), [vectors])[0]
    return quaternion_polynomial(lmax, False, [q])
