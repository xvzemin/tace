"""CUDA channel-preserving path sums in either storage layout."""

import torch

from ....utils.metadata import parse_metadata
from ...precision import validate_uniform
from ..torch.uu_linear import UuLinear, supported
from .pointwise import evaluate as pointwise


@torch.library.custom_op("eqx::uu_linear", mutates_args=(), device_types="cuda")
def evaluate(
    metadata: str, features: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Fuse copy contractions, normalization and output storage."""
    validate_uniform((features, weight))
    if not supported(features, weight):
        raise ValueError("UuLinear requires matrices and shared or per-row weights.")
    storage, inputs, outputs, channels, mode, paths = parse_metadata(metadata)
    din, dout = sum(d * m for d, m in inputs), sum(d * m for d, m in outputs)
    nw = sum(ni * no * channels for _, _, _, ni, no, _, _ in paths)
    if storage not in ("ir_mul", "mul_ir") or mode not in ("sum", "expand"):
        raise ValueError("Invalid UuLinear storage or path mode.")
    if features.shape[-1] != din or weight.shape[-1] != nw:
        raise ValueError("Feature or weight dimension does not match the paths.")
    result = features.new_empty((features.shape[0], dout))
    if not result.numel():
        return result
    shared = weight.ndim == 1 or weight.shape[0] == 1
    offsets = [sum(d * m for d, m in inputs[:i]) for i in range(len(inputs))]
    cases, start = [], 0
    for j, (dim, mul) in enumerate(outputs):
        if not dim * mul:
            continue
        component = (
            f"(c-{start})%{dim}" if storage == "mul_ir" else f"(c-{start})/{mul}"
        )
        copy = f"(c-{start})/{dim}" if storage == "mul_ir" else f"(c-{start})%{mul}"
        terms = []
        for index, destination, sw, ni, no, scale, expanded in paths:
            if destination != j:
                continue
            address = (
                f"u*{channels * dim}+ch*{dim}+m"
                if storage == "mul_ir"
                else f"m*{inputs[index][1]}+u*{channels}+ch"
            )
            expression = f"p0[row*{features.stride(0)}+({offsets[index]}+{address})*{features.stride(1)}]*(p1[wr*{0 if weight.ndim == 1 else weight.stride(0)}+({sw}+(u*{no}+v)*{channels}+ch)*{weight.stride(-1)}]*T({scale:.17g}))"
            if mode == "expand":
                terms.append(f"""
                    if (q>={expanded} && q<{expanded + ni * no * channels}) {{
                        int u=(q-{expanded})/{no * channels};
                        int v=((q-{expanded})/{channels})%{no};
                        total += {expression};
                    }}
                """)
            else:
                terms.append(f"""
                    {{ int v=q/{channels};
                        for(int u=0;u<{ni};++u) total += {expression};
                    }}
                """)
        cases.append(f"""
            if(c>={start} && c<{start + dim * mul}) {{
                int m={component},q={copy},ch=q%{channels}; T total=0;
                {"".join(terms)}
                out[i]=total;
            }}
        """)
        start += dim * mul
    return pointwise(
        result,
        (features, weight),
        f"I row=i/{dout},wr={'0' if shared else 'row'}; int c=i%{dout}; {''.join(cases)}",
    )


@evaluate.register_fake
def fake(metadata, features, weight):
    _, _, outputs, _, _, _ = parse_metadata(metadata)
    return features.new_empty((features.shape[0], sum(d * m for d, m in outputs)))


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, features, weight = inputs
    ctx.save_for_backward(features, weight)


evaluate.register_autograd(UuLinear.backward, setup_context=setup_context)
