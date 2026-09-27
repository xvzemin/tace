"""Indexed feature sums and their transposes."""

from functools import lru_cache

import torch

from ..utils import parse_metadata
from .cuda import kernels, runtime


@lru_cache(maxsize=128)
def permutation_source(widths, dtype, transpose):
    """Transpose the batch and group axes without expanding feature entries."""
    scalar = "double" if dtype == torch.float64 else "float"
    total = sum(widths)
    lines, start = [], 0
    for width in widths:
        if not width:
            continue
        end = start + width
        if transpose:
            condition = f"col < {end}"
            address = f"rows * {start} + row * {width} + col - {start}"
        else:
            condition = f"index < rows * {end}"
            address = f"((index - rows * {start}) / {width}) * {total} + indices[{start} + (index - rows * {start}) % {width}]"
        lines.append(f"if ({condition}) {{ address = {address}; }}")
        start = end
    select = " else ".join(lines)
    return f"""
    using T = {scalar};
    using int64_t = long long;
    extern "C" __global__ void run(const T* input, T* output,
        const int64_t* indices, int64_t rows, int64_t s0, int64_t s1) {{
        const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= rows * {total}) return;
        const int64_t row = index / {max(total, 1)};
        const int64_t col = indices[index % {max(total, 1)}];
        int64_t address = 0;
        {select}
        output[index] = input[(address / {max(total, 1)}) * s0 + (address % {max(total, 1)}) * s1];
    }}
    """


@torch.library.custom_op("eqx::grouped_permute", mutates_args=(), device_types="cuda")
def grouped_permute(
    features: torch.Tensor,
    indices: torch.Tensor,
    inverse: torch.Tensor,
    widths: str,
    transpose: bool = False,
) -> torch.Tensor:
    """Pack feature groups into contiguous batch matrices, or apply its transpose."""
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Grouped permutations require float32 or float64 features.")
    result = torch.empty_like(features, memory_format=torch.contiguous_format)
    if result.numel():
        code = permutation_source(parse_metadata(widths), features.dtype, transpose)
        kernel = kernels([code], features.device)[code]
        runtime().launch(
            [
                (
                    kernel,
                    [
                        features.data_ptr(),
                        result.data_ptr(),
                        indices.data_ptr(),
                        features.size(0),
                        *features.stride(),
                    ],
                    (result.numel() + 255) // 256,
                    1,
                    256,
                    0,
                )
            ],
            torch.cuda.current_stream(features.device).cuda_stream,
        )
    return result


@grouped_permute.register_fake
def grouped_permute_fake(features, indices, inverse, widths, transpose=False):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


def permutation_setup_context(ctx, inputs, output):
    _, indices, inverse, ctx.widths, ctx.transpose = inputs
    ctx.save_for_backward(inverse, indices)


def permutation_backward(ctx, gradient):
    inverse, indices = ctx.saved_tensors
    return (
        grouped_permute(gradient, inverse, indices, ctx.widths, not ctx.transpose),
        None,
        None,
        None,
        None,
    )


grouped_permute.register_autograd(
    permutation_backward, setup_context=permutation_setup_context
)


@lru_cache(maxsize=6)
def sum_source(dtype, normalization):
    scalar = "double" if dtype == torch.float64 else "float"
    value = "input[row * s0 + indices[k] * s1]"
    if normalization == 2:
        value += " / T(count[indices[k]])"
    result = "value / T(count[col])" if normalization == 1 else "value"
    return f"""
    using T = {scalar};
    using int64_t = long long;
    extern "C" __global__ void run(const T* input, T* output,
        const int64_t* indices, const int64_t* ptr, const int64_t* count,
        int64_t rows, int64_t width, int64_t s0, int64_t s1) {{
        const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= rows * width) return;
        const int64_t row = index / width, col = index % width;
        T value = 0;
        for (int64_t k = ptr[col]; k < ptr[col + 1]; ++k)
            value += {value};
        output[index] = {result};
    }}
    """


@torch.library.custom_op("eqx::indexed_sum", mutates_args=(), device_types="cuda")
def indexed_sum(
    features: torch.Tensor,
    indices: torch.Tensor,
    ptr: torch.Tensor,
    transpose_indices: torch.Tensor,
    transpose_ptr: torch.Tensor,
    count: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
    """Sum indexed columns without an expanded gather or atomic additions.

    Parameters
    ----------
    features : torch.Tensor
        Input matrix of shape ``(batch, features)``.
    indices, ptr : torch.Tensor
        Column indices and row pointers of the binary CSR map.
    transpose_indices, transpose_ptr : torch.Tensor
        CSR indices of its transpose, used by recursive backward passes.
    count : torch.Tensor
        Row divisors, or an empty tensor for unnormalized sums.
    transpose : bool, optional
        Apply divisors to inputs instead of outputs in a transposed map.
    """
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Indexed sums require float32 or float64 features.")
    result = indexed_sum_fake(
        features, indices, ptr, transpose_indices, transpose_ptr, count, transpose
    )
    if result.numel():
        code = sum_source(
            features.dtype, (2 if transpose else 1) if count.numel() else 0
        )
        kernel = kernels([code], features.device)[code]
        runtime().launch(
            [
                (
                    kernel,
                    [
                        features.data_ptr(),
                        result.data_ptr(),
                        indices.data_ptr(),
                        ptr.data_ptr(),
                        count.data_ptr(),
                        features.size(0),
                        result.size(1),
                        *features.stride(),
                    ],
                    (result.numel() + 255) // 256,
                    1,
                    256,
                    0,
                )
            ],
            torch.cuda.current_stream(features.device).cuda_stream,
        )
    return result


@indexed_sum.register_fake
def indexed_sum_fake(
    features, indices, ptr, transpose_indices, transpose_ptr, count, transpose=False
):
    return features.new_empty((features.size(0), ptr.numel() - 1))


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:-1])
    ctx.transpose = inputs[-1]


def backward(ctx, gradient):
    indices, ptr, transpose_indices, transpose_ptr, count = ctx.saved_tensors
    return (
        indexed_sum(
            gradient,
            transpose_indices,
            transpose_ptr,
            indices,
            ptr,
            count,
            not ctx.transpose,
        ),
        None,
        None,
        None,
        None,
        None,
        None,
    )


indexed_sum.register_autograd(backward, setup_context=setup_context)
