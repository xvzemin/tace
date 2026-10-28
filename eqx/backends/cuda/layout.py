"""CUDA permutations, gathers and indexed reductions."""

from functools import lru_cache

import torch

from ...ops.layout import gather_sum_fake, indexed_sum_fake
from ...utils.metadata import parse_metadata
from .runtime import kernels, runtime


@lru_cache(maxsize=128)
def gather_sum_source(dtype, indexed):
    scalar = "double" if dtype == torch.float64 else "float"
    arguments, values = [], []
    for i, gather in enumerate(indexed):
        arguments.extend((f"const T* input{i}", f"int64_t s{i}0", f"int64_t s{i}1"))
        if gather:
            arguments.append(f"const int64_t* index{i}")
        row = f"index{i}[row]" if gather else "row"
        values.append(f"input{i}[({row}) * s{i}0 + col * s{i}1]")
    accumulation = "\n".join(f"value += {value};" for value in values[1:])
    return f"""
    using T = {scalar};
    using int64_t = long long;
    extern "C" __global__ void run({", ".join(arguments)},
        T* output, int64_t rows, int64_t width) {{
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= rows * width) return;
        const int64_t row = i / width, col = i % width;
        T value = {values[0]};
        {accumulation}
        output[i] = value;
    }}
    """


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


def gather_sum(
    features: list[torch.Tensor],
    indices: list[torch.Tensor | None],
) -> torch.Tensor:
    """Sum feature matrices after optional row gathers.

    Parameters
    ----------
    features : list of torch.Tensor
        Matrices with the same channel width, dtype and device.
    indices : list of torch.Tensor or None
        Row indices for each matrix. ``None`` leaves its rows unchanged.
    """
    if not features or len(features) != len(indices):
        raise ValueError("Each feature matrix requires an index tensor or None.")
    if features[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Gather sums require float32 or float64 features.")
    for value, index in zip(features, indices):
        if (
            value.ndim != 2
            or value.dtype != features[0].dtype
            or value.device != features[0].device
        ):
            raise ValueError("Feature matrices must have the same dtype and device.")
        if index is not None and (
            index.dtype != torch.int64
            or index.ndim != 1
            or index.device != value.device
        ):
            raise ValueError(
                "Row indices must be one-dimensional int64 tensors on the feature device."
            )
    result = gather_sum_fake(features, indices)
    for value, index in zip(features, indices):
        if value.size(1) != result.size(1) or (
            value.size(0) if index is None else index.numel()
        ) != result.size(0):
            raise ValueError("Gathered feature matrices must have the same shape.")
    if result.numel():
        indices = [None if index is None else index.contiguous() for index in indices]
        code = gather_sum_source(
            features[0].dtype, tuple(i is not None for i in indices)
        )
        kernel = kernels([code], result.device)[code]
        arguments = []
        for value, index in zip(features, indices):
            arguments.extend((value.data_ptr(), *value.stride()))
            if index is not None:
                arguments.append(index.data_ptr())
        arguments.extend((result.data_ptr(), *result.shape))
        runtime().launch(
            [(kernel, arguments, (result.numel() + 255) // 256, 1, 256, 0)],
            torch.cuda.current_stream(result.device).cuda_stream,
        )
    return result


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
