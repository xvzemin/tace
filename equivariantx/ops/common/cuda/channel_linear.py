"""Native shared channel GEMMs with fused path scaling, summation and layout."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils.metadata import parse_metadata
from ...precision import validate_uniform
from ..torch.channel_linear import ChannelLinear


@triton.jit
def channel_kernel(
    features,
    weight,
    bias,
    output,
    rows,
    INPUTS: tl.constexpr,
    PATHS: tl.constexpr,
    INPUT_OFFSETS: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr,
    COMPONENTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    OUTPUT_INDEX: tl.constexpr,
    BIAS_OFFSET: tl.constexpr,
    CARTESIAN: tl.constexpr,
    PRECISION: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    node, component = r // COMPONENTS, r % COMPONENTS
    k = tl.arange(0, BK)
    accumulator = tl.zeros(
        (BM, BN), tl.float64 if output.dtype.element_ty == tl.float64 else tl.float32
    )
    for p in tl.static_range(len(PATHS)):
        if PATHS[p][1] == OUTPUT_INDEX:
            ci = INPUTS[PATHS[p][0]][1]
            input_offset = INPUT_OFFSETS[PATHS[p][0]]
            for block in range((ci + BK - 1) // BK):
                channel = block * BK + k
                if CARTESIAN:
                    address = (
                        node[:, None] * INPUT_DIM
                        + input_offset
                        + channel[None, :] * COMPONENTS
                        + component[:, None]
                    )
                else:
                    address = (
                        node[:, None] * INPUT_DIM
                        + input_offset
                        + component[:, None] * ci
                        + channel[None, :]
                    )
                x = tl.load(
                    features + address,
                    (node[:, None] < rows) & (channel[None, :] < ci),
                    0,
                )
                w = tl.load(
                    weight + PATHS[p][2] + channel[:, None] * CHANNELS + c[None, :],
                    (channel[:, None] < ci) & (c[None, :] < CHANNELS),
                    0,
                )
                w = w * PATHS[p][3]
                accumulator = tl.dot(x, w, accumulator, input_precision=PRECISION)
    if BIAS_OFFSET >= 0:
        accumulator += tl.load(bias + BIAS_OFFSET + c, c < CHANNELS, 0)[None, :]
    if CARTESIAN:
        address = (
            node[:, None] * OUTPUT_DIM
            + OUTPUT_OFFSET
            + c[None, :] * COMPONENTS
            + component[:, None]
        )
    else:
        address = (
            node[:, None] * OUTPUT_DIM
            + OUTPUT_OFFSET
            + component[:, None] * CHANNELS
            + c[None, :]
        )
    tl.store(
        output + address, accumulator, (node[:, None] < rows) & (c[None, :] < CHANNELS)
    )


@triton_op("eqx::channel_linear", mutates_args=())
def evaluate(
    metadata: str, features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Evaluate shared channel maps with FP32/TF32 or FP64 matrix arithmetic."""
    validate_uniform((features, weight, bias))
    if not features.is_cuda or features.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError(
            "Native channel maps require CUDA matrices and shared parameter vectors."
        )
    storage, (inputs, outputs, paths, biases, groups) = parse_metadata(metadata)
    features, weight, bias = (
        features.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
    )
    din = sum(dim * mul for dim, mul in inputs)
    dout = sum(dim * mul for dim, mul in outputs)
    expected_weight = sum(inputs[i][1] * outputs[j][1] for i, j, start, scale in paths)
    expected_bias = sum(size for j, start, size in biases)
    if storage not in ("ir_mul", "mul_ir") or features.shape[1] != din:
        raise ValueError("Invalid channel layout or feature dimension.")
    if weight.numel() != expected_weight or bias.numel() != expected_bias:
        raise ValueError("Parameter dimensions do not match the channel map.")
    result = features.new_empty((features.shape[0], dout))
    if dout:
        input_offsets = tuple(
            sum(dim * mul for dim, mul in inputs[:i]) for i in range(len(inputs))
        )
        offset = 0
        precision = "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"
        for j, (dim, mul) in enumerate(outputs):
            if dim * mul:
                bias_offset = next(
                    (start for dest, start, size in biases if dest == j), -1
                )
                tf32 = features.dtype == torch.float32 and precision == "tf32"
                wide = tf32 and storage == "ir_mul" and mul >= 1024
                bm, bn = (128 if wide else 32), (64 if tf32 and mul >= 64 else 32)
                wrap_triton(channel_kernel)[
                    (triton.cdiv(features.shape[0] * dim, bm), triton.cdiv(mul, bn))
                ](
                    features,
                    weight,
                    bias,
                    result,
                    features.shape[0],
                    inputs,
                    paths,
                    input_offsets,
                    din,
                    dout,
                    offset,
                    dim,
                    mul,
                    j,
                    bias_offset,
                    storage == "mul_ir",
                    precision,
                    bm,
                    bn,
                    32,
                    num_warps=8 if wide else 4,
                )
            offset += dim * mul
    return result


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, features, weight, bias = inputs
    ctx.bias_numel = bias.numel()
    ctx.save_for_backward(features, weight)


evaluate.register_autograd(ChannelLinear.backward, setup_context=setup_context)
