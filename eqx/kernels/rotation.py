"""Degree-wise frame rotations and their recursive transposes."""

from ast import literal_eval
from functools import lru_cache

import torch

from .cuda import kernels, runtime

parse = lru_cache(maxsize=128)(literal_eval)


def rotate(features, matrix, metadata):
    """Multiply frames within matching degrees without dense angular work."""
    empty = features.new_empty((0, matrix.shape[1], features.shape[2]))
    return contract(features, matrix, empty, metadata, 2)


@lru_cache(maxsize=128)
def source(dtype, channels, metadata, role):
    degrees_in, degrees_out = parse(metadata)
    ni, no = len(degrees_in), len(degrees_out)
    scalar = "double" if dtype == torch.float64 else "float"
    groups = []
    if role == 1:
        pairs = [
            (i, j)
            for i, a in enumerate(degrees_out)
            for j, b in enumerate(degrees_in)
            if a == b
        ]
        rows = ",".join(str(i) for i, _ in pairs)
        columns = ",".join(str(j) for _, j in pairs)
        body = f"""
        long long e=blockIdx.x; int p=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x%32;
        if(p>={len(pairs)}) return;
        int i=rows[p],j=columns[p]; scalar sum=0;
        for(int c=lane;c<{channels};c+=32)
            sum+=features[(e*{ni}+j)*{channels}+c]*edge[(e*{no}+i)*{channels}+c];
        for(int offset=16;offset;offset>>=1) sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(!lane) result[(e*{no}+i)*{ni}+j]=sum;
        """
        constants = (
            f"__device__ __constant__ int rows[]={{{rows}}},columns[]={{{columns}}};"
        )
        grid = (len(pairs) + 3) // 4
    else:
        outputs, inputs = (
            (degrees_out, degrees_in) if role == 2 else (degrees_in, degrees_out)
        )
        for i, degree in enumerate(outputs):
            terms = []
            for j, other in enumerate(inputs):
                if degree != other:
                    continue
                if role == 2:
                    terms.append(
                        f"sum+=matrix[(e*{no}+{i})*{ni}+{j}]*features[(e*{ni}+{j})*{channels}+c];"
                    )
                else:
                    terms.append(
                        f"sum+=matrix[(e*{no}+{j})*{ni}+{i}]*edge[(e*{no}+{j})*{channels}+c];"
                    )
            groups.append(f"case {i}: {{ {''.join(terms)} break; }}")
        constants = ""
        body = f"""
        long long e=blockIdx.x; int ac=blockIdx.y*128+threadIdx.x;
        if(ac>={len(outputs) * channels}) return;
        int a=ac/{channels},c=ac%{channels}; scalar sum=0;
        switch(a) {{ {"".join(groups)} }}
        result[e*{len(outputs) * channels}+ac]=sum;
        """
        grid = (len(outputs) * channels + 127) // 128
    return (
        f"""
    using scalar={scalar};
    {constants}
    extern "C" __global__ void run(const scalar* features,const scalar* matrix,
        const scalar* edge,scalar* result) {{ {body} }}
    """,
        grid,
    )


@torch.library.custom_op("eqx::frame_rotation", mutates_args=(), device_types="cuda")
def contract(
    features: torch.Tensor,
    matrix: torch.Tensor,
    edge: torch.Tensor,
    metadata: str,
    role: int,
) -> torch.Tensor:
    features, matrix, edge = (x.contiguous() for x in (features, matrix, edge))
    result = contract_fake(features, matrix, edge, metadata, role)
    if role == 1:
        result.zero_()
    if features.shape[0]:
        code, groups = source(features.dtype, features.shape[2], metadata, role)
        compiled = kernels([code], features.device)
        args = [x.data_ptr() for x in (features, matrix, edge, result)]
        runtime().launch(
            [(compiled[code], args, features.shape[0], groups, 128, 0)],
            torch.cuda.current_stream(features.device).cuda_stream,
        )
    return result


@contract.register_fake
def contract_fake(features, matrix, edge, metadata, role):
    if role == 0:
        return torch.empty_like(features)
    if role == 1:
        return torch.empty_like(matrix)
    return features.new_empty((features.shape[0], matrix.shape[1], features.shape[2]))


def setup_context(ctx, inputs, output):
    features, matrix, edge, ctx.kernel_metadata, ctx.role = inputs
    ctx.save_for_backward(features, matrix, edge)


def backward(ctx, grad):
    values = list(ctx.saved_tensors)
    values[ctx.role] = grad
    return (
        *[
            contract(*values, ctx.kernel_metadata, role)
            if role != ctx.role and ctx.needs_input_grad[role]
            else None
            for role in range(3)
        ],
        None,
        None,
    )


contract.register_autograd(backward, setup_context=setup_context)
