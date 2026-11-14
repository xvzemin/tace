"""Compiler-visible channel-preserving path sums in either storage layout."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton

from ....utils.metadata import parse_metadata
from ...precision import validate_uniform
from ..torch.uu_linear import UuLinear, supported


@triton.jit
def path_kernel(
    features,
    weight,
    output,
    rows,
    INPUTS: tl.constexpr,
    PATHS: tl.constexpr,
    OFFSETS: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    WEIGHT_DIM: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr,
    OUTPUT_INDEX: tl.constexpr,
    COMPONENTS: tl.constexpr,
    MULTIPLICITY: tl.constexpr,
    CHANNELS: tl.constexpr,
    CARTESIAN: tl.constexpr,
    EXPAND: tl.constexpr,
    SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, within = (
        index // (COMPONENTS * MULTIPLICITY),
        index % (COMPONENTS * MULTIPLICITY),
    )
    if CARTESIAN:
        component, mul = within % COMPONENTS, within // COMPONENTS
    else:
        component, mul = within // MULTIPLICITY, within % MULTIPLICITY
    channel = mul % CHANNELS
    weight_row = 0 if SHARED else row
    value = tl.full(
        (BLOCK,), 0, tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32
    )
    for p in tl.static_range(len(PATHS)):
        if PATHS[p][1] == OUTPUT_INDEX:
            ni, no = PATHS[p][3], PATHS[p][4]
            if EXPAND:
                local = mul - PATHS[p][6]
                valid = (row < rows) & (local >= 0) & (local < ni * no * CHANNELS)
                copy_in = local // (no * CHANNELS)
                copy_out = (local // CHANNELS) % no
                input_mul = copy_in * CHANNELS + channel
                if CARTESIAN:
                    address = input_mul * COMPONENTS + component
                else:
                    address = component * INPUTS[PATHS[p][0]][1] + input_mul
                x = tl.load(
                    features + row * INPUT_DIM + OFFSETS[PATHS[p][0]] + address,
                    valid,
                    0,
                )
                w = tl.load(
                    weight
                    + weight_row * WEIGHT_DIM
                    + PATHS[p][2]
                    + (copy_in * no + copy_out) * CHANNELS
                    + channel,
                    valid,
                    0,
                )
                value += x * (w * PATHS[p][5])
            else:
                copy_out = mul // CHANNELS
                for copy_in in tl.static_range(PATHS[p][3]):
                    input_mul = copy_in * CHANNELS + channel
                    if CARTESIAN:
                        address = input_mul * COMPONENTS + component
                    else:
                        address = component * INPUTS[PATHS[p][0]][1] + input_mul
                    x = tl.load(
                        features + row * INPUT_DIM + OFFSETS[PATHS[p][0]] + address,
                        row < rows,
                        0,
                    )
                    w = tl.load(
                        weight
                        + weight_row * WEIGHT_DIM
                        + PATHS[p][2]
                        + (copy_in * no + copy_out) * CHANNELS
                        + channel,
                        row < rows,
                        0,
                    )
                    value += x * (w * PATHS[p][5])
    tl.store(output + row * OUTPUT_DIM + OUTPUT_OFFSET + within, value, row < rows)


@triton_op("eqx::uu_linear", mutates_args=())
def evaluate(
    metadata: str, features: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Fuse copy contractions, normalization and output storage without channel GEMMs."""
    validate_uniform((features, weight))
    if not features.is_cuda or not supported(features, weight):
        raise ValueError(
            "Native UuLinear requires CUDA node matrices and shared or per-node weights."
        )
    storage, inputs, outputs, channels, mode, paths = parse_metadata(metadata)
    din, dout = sum(d * m for d, m in inputs), sum(d * m for d, m in outputs)
    weight_dim = sum(
        ni * no * channels for i, j, start, ni, no, scale, expanded in paths
    )
    if storage not in ("ir_mul", "mul_ir") or mode not in ("sum", "expand"):
        raise ValueError("Invalid UuLinear storage or path mode.")
    if features.shape[-1] != din or weight.shape[-1] != weight_dim:
        raise ValueError(
            "Feature or weight dimension does not match the UuLinear paths."
        )
    features, weight = features.contiguous(), weight.contiguous()
    result = features.new_empty((features.shape[0], dout))
    if dout:
        offsets = tuple(sum(d * m for d, m in inputs[:i]) for i in range(len(inputs)))
        shared = weight.ndim == 1 or statically_known_true(weight.shape[0] == 1)
        offset = 0
        for j, (dim, mul) in enumerate(outputs):
            if dim * mul:
                wrap_triton(path_kernel)[
                    (triton.cdiv(features.shape[0] * dim * mul, 256),)
                ](
                    features,
                    weight,
                    result,
                    features.shape[0],
                    inputs,
                    paths,
                    offsets,
                    din,
                    dout,
                    weight_dim,
                    offset,
                    j,
                    dim,
                    mul,
                    channels,
                    storage == "mul_ir",
                    mode == "expand",
                    shared,
                    256,
                )
            offset += dim * mul
    return result


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, features, weight = inputs
    ctx.save_for_backward(features, weight)


evaluate.register_autograd(UuLinear.backward, setup_context=setup_context)
