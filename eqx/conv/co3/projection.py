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


@lru_cache(maxsize=128)
def packing_source(metadata, dtype, transpose):
    """Combine an input permutation with blockwise angular extraction."""
    input_dim, groups = parse_metadata(metadata)
    scalar = "double" if dtype == torch.float64 else "float"
    total = sum(g[1] * g[3] for g in groups)
    cases = []
    for (
        offset,
        channels,
        dim_in,
        dim_out,
        coefficients,
        channel,
        width,
        start,
    ) in groups:
        rows = [[] for _ in range(dim_in if transpose else dim_out)]
        for a, b, value in coefficients:
            if transpose:
                address = f"rows*{start}+(n*{dim_out}+{b})*{width}+{channel}+u"
                term = f"input[({address})/{max(total, 1)}*s0+({address})%{max(total, 1)}*s1]"
                rows[a].append(f"scalar({value:.17g})*{term}")
            else:
                rows[b].append(
                    f"scalar({value:.17g})*input[n*s0+indices[{offset + a * channels}+u]*s1]"
                )
        for a, terms in enumerate(rows):
            address = (
                f"n*{input_dim}+indices[{offset + a * channels}+u]"
                if transpose
                else f"rows*{start}+(n*{dim_out}+{a})*{width}+{channel}+u"
            )
            value = "+".join(terms) if terms else "scalar(0)"
            cases.append(
                f"case {len(cases)}: {{ if(index>=rows*{channels}) return; "
                f"long long n=index/{channels}; int u=index%{channels}; "
                f"output[{address}]={value}; return; }}"
            )
    return (
        f"""
        using scalar={scalar};
        extern "C" __global__ void run(const scalar* input, const long long* indices,
            scalar* output, long long rows, long long s0, long long s1) {{
            long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            switch(blockIdx.y) {{ {"".join(cases)} }}
        }}
    """,
        max((g[1] for g in groups), default=0),
        len(cases),
    )


@torch.library.custom_op("eqx::cartesian_pack", mutates_args=(), device_types="cuda")
def pack(
    metadata: str,
    features: torch.Tensor,
    indices: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
    """Extract angular coordinates into contiguous channel-last blocks.

    ``indices`` must be a permutation of the input columns. The transpose
    restores their original layout without atomic reductions.
    """
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Cartesian packing requires float32 or float64 features.")
    indices = indices.contiguous()
    output = pack_fake(metadata, features, indices, transpose)
    if output.numel():
        code, channels, count = packing_source(metadata, features.dtype, transpose)
        if count:
            kernel = kernels([code], features.device)[code]
            runtime().launch(
                [
                    (
                        kernel,
                        [
                            features.data_ptr(),
                            indices.data_ptr(),
                            output.data_ptr(),
                            features.size(0),
                            *features.stride(),
                        ],
                        (features.size(0) * channels + 127) // 128,
                        count,
                        128,
                        0,
                    )
                ],
                torch.cuda.current_stream(features.device).cuda_stream,
            )
    return output


@pack.register_fake
def pack_fake(metadata, features, indices, transpose=False):
    input_dim, groups = parse_metadata(metadata)
    width = input_dim if transpose else sum(g[1] * g[3] for g in groups)
    return features.new_empty((features.size(0), width))


def packing_setup_context(ctx, inputs, output):
    ctx.kernel_metadata, _, indices, ctx.transpose = inputs
    ctx.save_for_backward(indices)


def packing_backward(ctx, gradient):
    (indices,) = ctx.saved_tensors
    return (
        None,
        pack(ctx.kernel_metadata, gradient, indices, not ctx.transpose),
        None,
        None,
    )


pack.register_autograd(packing_backward, setup_context=packing_setup_context)
