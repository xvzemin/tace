"""CUDA shared channel maps with fused path scaling, layout and bias."""

from functools import lru_cache

import torch

from ....utils.autograd import forward_ad_active
from ....utils.metadata import parse_metadata
from ...precision import validate_uniform
from ..torch.channel_linear import ChannelLinear
from .pointwise import evaluate as pointwise


@lru_cache(maxsize=128)
def scale_expression(blocks):
    """Generate one piecewise-constant normalization for a parameter vector."""
    start, terms = 0, []
    for width, scale in blocks:
        start += width
        terms.append((start, f"T({scale:.17g})"))
    expression = " : ".join(f"i < {end} ? {value}" for end, value in terms[:-1])
    return (
        "out[i] = p0[i] * ("
        + (expression + " : " if expression else "")
        + terms[-1][1]
        + ");"
    )


@torch.library.custom_op("eqx::path_scale", mutates_args=(), device_types="cuda")
def scale_cuda(metadata: str, weight: torch.Tensor) -> torch.Tensor:
    """Scale contiguous parameter blocks without storing a coefficient array."""
    validate_uniform((weight,))
    blocks = parse_metadata(metadata)
    if not blocks or weight.numel() != sum(n for n, _ in blocks):
        raise ValueError("Parameter blocks must cover the weight vector.")
    return pointwise(
        weight.new_empty(weight.shape), (weight.contiguous(),), scale_expression(blocks)
    )


@scale_cuda.register_fake
def scale_fake(metadata, weight):
    return weight.new_empty(weight.shape)


def setup_scale(ctx, inputs, output):
    ctx.metadata_key = inputs[0]


def scale_backward(ctx, gradient):
    return None, scale_cuda(ctx.metadata_key, gradient)


scale_cuda.register_autograd(scale_backward, setup_context=setup_scale)


def scale_parameters(weight, blocks):
    """Normalize shared path weights once before the edge program."""
    if not blocks or len(blocks) == 1 and blocks[0][1] == 1:
        return weight
    if len(blocks) == 1:
        return weight * blocks[0][1]
    if forward_ad_active():
        parts = weight.reshape(-1).split([n for n, _ in blocks])
        return torch.cat(
            [part * scale for part, (_, scale) in zip(parts, blocks)]
        ).reshape_as(weight)
    return scale_cuda(repr(blocks), weight)


@torch.library.custom_op("eqx::channel_linear", mutates_args=(), device_types="cuda")
def evaluate(
    metadata: str, features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Apply independent shared channel maps without expanded coefficients."""
    validate_uniform((features, weight, bias))
    if features.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("Channel maps require matrices and shared parameter vectors.")
    storage, (inputs, outputs, paths, biases, groups) = parse_metadata(metadata)
    din = sum(dim * mul for dim, mul in inputs)
    dout = sum(dim * mul for dim, mul in outputs)
    if storage not in ("ir_mul", "mul_ir") or features.shape[-1] != din:
        raise ValueError("Invalid channel layout or feature dimension.")
    if weight.numel() != sum(inputs[i][1] * outputs[j][1] for i, j, _, _ in paths):
        raise ValueError("Invalid channel weight dimension.")
    if bias.numel() != sum(size for _, _, size in biases):
        raise ValueError("Invalid channel bias dimension.")
    result = features.new_empty((features.shape[0], dout))
    if not result.numel():
        return result
    offsets = [sum(d * m for d, m in inputs[:i]) for i in range(len(inputs))]
    cases, start = [], 0
    for j, (dim, co) in enumerate(outputs):
        if not dim * co:
            continue
        component = (
            f"(c-{start}) % {dim}" if storage == "mul_ir" else f"(c-{start}) / {co}"
        )
        channel = (
            f"(c-{start}) / {dim}" if storage == "mul_ir" else f"(c-{start}) % {co}"
        )
        terms = []
        for index, destination, sw, scale in paths:
            if destination != j:
                continue
            ci = inputs[index][1]
            pos = f"u*{dim}+m" if storage == "mul_ir" else f"m*{ci}+u"
            terms.append(f"""
                for (int u=0; u<{ci}; ++u)
                    total += p0[row*{features.stride(0)}+({offsets[index]}+{pos})*{features.stride(1)}]
                        * (p1[({sw}+u*{co}+v)*{weight.stride(0)}] * T({scale:.17g}));
            """)
        for destination, offset, size in biases:
            if destination == j:
                terms.append(f"total += p2[({offset}+v)*{bias.stride(0)}];")
        cases.append(f"""
            if (c>={start} && c<{start + dim * co}) {{
                int m={component}, v={channel}; T total=0;
                {"".join(terms)}
                out[i]=total;
            }}
        """)
        start += dim * co
    return pointwise(
        result,
        (features, weight, bias),
        f"I row=i/{dout}; int c=i%{dout}; {''.join(cases)}",
    )


@evaluate.register_fake
def fake(metadata, features, weight, bias):
    storage, (_, outputs, _, _, _) = parse_metadata(metadata)
    return features.new_empty((features.shape[0], sum(d * m for d, m in outputs)))


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, features, weight, bias = inputs
    ctx.bias_numel = bias.numel()
    ctx.save_for_backward(features, weight)


evaluate.register_autograd(ChannelLinear.backward, setup_context=setup_context)
