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


def silu_dual(x, tangent, scale, grad=None, grad_tangent=None):
    """Apply a scaled SiLU JVP or reuse private GEMM adjoints for the reverse pass."""
    scalar = scalar_type(x.dtype)
    args = [x]
    if grad is None:
        args.append(tangent)
        primal = "v * s"
        dual = "first * p1[i]"
    else:
        args.append(grad)
        primal = "first * p1[i]"
        parts = []
        if tangent is not None:
            parts.append(f"d * (T(2) + v * (T(1) - T(2)*s)) * p1[i] * p{len(args)}[i]")
            args.append(tangent)
        if grad_tangent is not None:
            parts.append(f"first * p{len(args)}[i]")
            args.append(grad_tangent)
        dual = " + ".join(parts) if parts else None
    inputs = tuple(v.contiguous() for v in args)
    a = torch.empty_like(inputs[0]) if grad is None else inputs[1]
    b = (
        None
        if dual is None
        else (torch.empty_like(inputs[0]) if grad_tangent is None else inputs[-1])
    )
    arguments = ", ".join(f"const T* p{k}" for k in range(len(inputs)))
    gain = f"T({float(scale).hex()})"
    derivative = "" if dual is None else f"T db = {gain} * ({dual});"
    second = "" if dual is None else "T* b,"
    source = f"""
    using T = {scalar};
    extern "C" __global__ void run({arguments}, T* a, {second} long long count) {{
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < count; i += (long long)gridDim.x * blockDim.x) {{
            T v = p0[i], s = T(1)/(T(1)+exp(-v)), d = s*(T(1)-s), first = s+v*d;
            T da = {gain} * ({primal});
            {derivative}
            a[i] = da;
            {"" if dual is None else "b[i] = db;"}
        }}
    }}
    """
    outputs = (a,) if b is None else (a, b)
    launch(source, (*inputs, *outputs), (a.numel(),), a.numel())
    return a, b


def coordinate_offset(shape, strides, row):
    """Map a valid flattened row without specializing its outer extent."""
    axes = [(n, s) for n, s in zip(shape, strides) if n != 1]
    terms, divisor = [], 1
    for axis in range(len(axes) - 1, -1, -1):
        size, stride = axes[axis]
        if stride:
            coordinate = f"({row}/{divisor})" if divisor != 1 else row
            if axis:
                coordinate = f"({coordinate}%{size})"
            terms.append(f"({coordinate}*{stride})")
        divisor *= size
    return " + ".join(terms) or "0"


def broadcast_offset(tensor, shape, trailing, *, row="row"):
    """Generate an offset for a broadcast row without expanding its storage."""
    leading = tensor.shape[:-trailing]
    strides = (0,) * (len(shape) - len(leading)) + tuple(
        0 if size == 1 else stride
        for size, stride in zip(leading, tensor.stride()[:-trailing])
    )
    return coordinate_offset(shape, strides, row)
