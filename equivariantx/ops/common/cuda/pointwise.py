"""CUDA source generation and launches for scalar tensor maps."""

import torch

from .runtime import kernels, runtime


def scalar_type(dtype):
    """Return the CUDA storage type for an operator argument."""
    return {
        torch.float32: "float",
        torch.float64: "double",
        torch.int8: "signed char",
        torch.uint8: "unsigned char",
        torch.int16: "short",
        torch.int32: "int",
        torch.int64: "long long",
        torch.bool: "bool",
    }[dtype]


def launch(source, tensors, scalars, count, *, threads=256, groups=1, shared=0):
    """Launch on the caller's CUDA stream, compiling only on first use."""
    if not count or not groups:
        return
    device = tensors[0].device
    if any(value.device != device for value in tensors):
        raise ValueError("CUDA operands must be on the same device.")
    kernel = kernels((source,), device)[source]
    runtime().launch(
        [
            (
                kernel,
                [value.data_ptr() for value in tensors] + list(scalars),
                min((count + threads - 1) // threads, 65535),
                groups,
                threads,
                shared,
            )
        ],
        torch.cuda.current_stream(device).cuda_stream,
    )


def evaluate(result, inputs, body, *, declarations=""):
    """Evaluate one scalar expression per output element."""
    arguments = ", ".join(
        f"const {scalar_type(value.dtype)}* __restrict__ p{k}"
        for k, value in enumerate(inputs)
    )
    source = f"""
    using T = {scalar_type(result.dtype)};
    using I = long long;
    {declarations}
    extern "C" __global__ void run({arguments}, T* __restrict__ out, I count) {{
        for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += I(gridDim.x) * blockDim.x) {{
            {body}
        }}
    }}
    """
    launch(source, (*inputs, result), (result.numel(),), result.numel())
    return result


def broadcast_offset(tensor, shape, trailing, *, row="row"):
    """Generate an offset for a broadcast row without expanding its storage."""
    leading = tensor.shape[:-trailing]
    strides = (0,) * (len(shape) - len(leading)) + tuple(
        0 if size == 1 else stride
        for size, stride in zip(leading, tensor.stride()[:-trailing])
    )
    terms, divisor = [], 1
    for size, stride in reversed(tuple(zip(shape, strides))):
        if stride:
            terms.append(f"(({row} / {divisor}) % {size}) * {stride}")
        divisor *= size
    return " + ".join(terms) or "0"
