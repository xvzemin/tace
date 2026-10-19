"""Fused Cartesian projections and their transposes."""

from collections import defaultdict
from functools import lru_cache

import torch

from ...kernels.cuda import kernels, runtime
from ...utils.metadata import parse_metadata


@lru_cache(maxsize=128)
def source(metadata, dtype, transpose):
    """Generate independent output rows of a fixed sparse linear map."""
    output_dim, groups = parse_metadata(metadata)
    scalar = "double" if dtype == torch.float64 else "float"
    arguments = [
        f"const scalar* p{i}, long long s{i}0, long long s{i}1, long long s{i}2"
        for i in range(1 if transpose else len(groups))
    ]
    arguments += [f"scalar* out{i}" for i in range(len(groups) if transpose else 1)]
    rows = defaultdict(list)
    for i, (offset, channels, dim_in, dim_out, coefficients) in enumerate(groups):
        for a, b, value in coefficients:
            key = (i, a, channels) if transpose else (offset, b, channels, dim_out)
            address = (
                f"n*s00+({offset}+u*{dim_out}+{b})*s01"
                if transpose
                else f"n*s{i}0+{a}*s{i}1+u*s{i}2"
            )
            rows[key].append(
                f"scalar({value:.17g})*p{0 if transpose else i}[{address}]"
            )
    cases = []
    written = [0] * (len(groups) if transpose else 1)
    for index, (key, terms) in enumerate(rows.items()):
        if transpose:
            i, a, channels = key
            address = f"out{i}[(n*{groups[i][2]}+{a})*{channels}+u]"
            written[i] += channels
        else:
            offset, b, channels, dim_out = key
            address = f"out0[n*{output_dim}+{offset}+u*{dim_out}+{b}]"
            written[0] += channels
        cases.append(
            f"case {index}: {{ if(index>=nodes*{channels}) return; "
            f"long long n=index/{channels}; int u=index%{channels}; "
            f"{address}=" + "+".join(terms) + "; return; }"
        )
    return (
        f"""using scalar={scalar};
        extern "C" __global__ void run({", ".join(arguments)}, long long nodes) {{
            long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            switch(blockIdx.y) {{ {"".join(cases)} }}
        }}""",
        max((group[1] for group in groups), default=0),
        len(rows),
        tuple(
            size == count
            for size, count in zip(
                [g[1] * g[2] for g in groups] if transpose else [output_dim], written
            )
        ),
    )


@torch.library.custom_op(
    "eqx::cartesian_projection", mutates_args=(), device_types="cuda"
)
def project(
    metadata: str, features: list[torch.Tensor], transpose: bool
) -> list[torch.Tensor]:
    """Project channel-last angular blocks into Cartesian mul_ir storage."""
    if features[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Cartesian projections require float32 or float64 features.")
    outputs = project_fake(metadata, features, transpose)
    if features[0].shape[0] == 0:
        return outputs
    code, width, count, complete = source(metadata, features[0].dtype, transpose)
    for output, full in zip(outputs, complete):
        if not full:
            output.zero_()
    if count:
        kernel = kernels([code], features[0].device)[code]
        arguments = []
        for value in features:
            arguments.extend((value.data_ptr(), *value.stride()))
            if transpose:
                arguments.append(0)
        arguments.extend(value.data_ptr() for value in outputs)
        arguments.append(features[0].shape[0])
        runtime().launch(
            [
                (
                    kernel,
                    arguments,
                    (features[0].shape[0] * width + 127) // 128,
                    count,
                    128,
                    0,
                )
            ],
            torch.cuda.current_stream(features[0].device).cuda_stream,
        )
    return outputs


@project.register_fake
def project_fake(metadata, features, transpose):
    output_dim, groups = parse_metadata(metadata)
    shapes = (
        [
            (features[0].shape[0], dim_in, channels)
            for _, channels, dim_in, _, _ in groups
        ]
        if transpose
        else [(features[0].shape[0], output_dim)]
    )
    return [features[0].new_empty(shape) for shape in shapes]


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, _, ctx.transpose = inputs


def backward(ctx, gradients):
    return None, project(ctx.kernel_metadata, gradients, not ctx.transpose), None


project.register_autograd(backward, setup_context=setup_context)
