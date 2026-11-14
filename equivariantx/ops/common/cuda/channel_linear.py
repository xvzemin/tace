"""CUDA shared channel maps with fused path scaling, layout and bias."""

import torch

from ....utils.metadata import parse_metadata
from ...precision import validate_uniform
from ..torch.channel_linear import ChannelLinear
from .pointwise import evaluate as pointwise


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
