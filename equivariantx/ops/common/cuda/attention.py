"""CUDA graph attention and derivatives."""

import struct
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton
from triton.language.extra.cuda import libdevice

from ....utils.autograd import forward_ad_active
from ....utils.outputs import attention_adjoint_outputs
from ...precision import validate_uniform
from .reduction import prepare_graph


@triton.jit
def maximum_kernel(
    scores, target, maximum, size, HEADS: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < size
    node = tl.load(target + index // HEADS, valid, 0)
    value = tl.load(scores + index, valid, -float("inf"))
    tl.atomic_max(maximum + node * HEADS + index % HEADS, value, valid, sem="relaxed")


@triton.jit
def normalize_kernel(
    scores,
    weight,
    target,
    maximum,
    denominator,
    output,
    size,
    HEADS: tl.constexpr,
    WEIGHT_HEADS: tl.constexpr,
    EPS: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < size
    edge, head = index // HEADS, index % HEADS
    node = tl.load(target + edge, valid, 0)
    offset = node * HEADS + head
    shift = tl.load(maximum + offset, valid, 0)
    shift = tl.where(shift == -float("inf"), 0, shift)
    value = tl.load(scores + index, valid, 0) - shift
    if scores.dtype.element_ty == tl.float64:
        value = libdevice.exp(value)
    else:
        value = tl.exp(value)
    if REDUCE:
        weight_head = head if WEIGHT_HEADS != 1 else 0
        multiplier = tl.load(weight + edge * WEIGHT_HEADS + weight_head, valid, 0)
        tl.atomic_add(denominator + offset, value * multiplier, valid, sem="relaxed")
    else:
        total = tl.load(denominator + offset, valid, 0)
        tl.store(output + index, value / (total + EPS), valid)


@triton_op("eqx::attention_normalized_exp", mutates_args=())
def contract(
    scores: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
    eps: float,
) -> torch.Tensor:
    """Normalize unsorted edges without an edge-sized exponential temporary."""
    validate_uniform((scores, weight))
    scores, weight, target = (value.contiguous() for value in (scores, weight, target))
    result = torch.empty_like(scores)
    if not statically_known_true(scores.numel() == 0):
        heads = scores.shape[1]
        maximum = scores.new_full((num_nodes, heads), -torch.inf)
        denominator = scores.new_zeros((num_nodes, heads))
        grid = (triton.cdiv(scores.numel(), 256),)
        wrap_triton(maximum_kernel)[grid](
            scores, target, maximum, scores.numel(), heads, 256
        )
        for reduce in (True, False):
            wrap_triton(normalize_kernel)[grid](
                scores,
                weight,
                target,
                maximum,
                denominator,
                result,
                scores.numel(),
                heads,
                weight.shape[1],
                eps,
                reduce,
                256,
            )
    return result


def contract_context(ctx, inputs, output):
    _, weight, target, ctx.num_nodes, _ = inputs
    ctx.save_for_backward(output, weight, target)


def contract_backward(ctx, gradient):
    values, weight, target = ctx.saved_tensors
    from ..torch.attention import evaluate_adjoint

    scores_grad, weight_grad, _ = evaluate_adjoint(
        values, gradient, weight, target, ctx.num_nodes
    )
    return scores_grad, weight_grad, None, None, None


contract.register_autograd(contract_backward, setup_context=contract_context)


def evaluate(scores, weight, target, num_nodes, eps):
    """Retain the native normalization in joint graphs and exact functional AD."""
    validate_uniform((scores, weight))
    if forward_ad_active() or not scores.is_cuda:
        from ..torch.attention import normalized_exp_reference

        return normalized_exp_reference(scores, weight, target, num_nodes, eps)
    return contract(scores, weight, target, num_nodes, eps)


@lru_cache(maxsize=32)
def merge_source(dtype, width, channels, heads, splits, eps):
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
    using scalar={scalar};
    extern "C" __global__ void run(const scalar* values,const scalar* denominators,
        const scalar* maxima,scalar* out,scalar* denominator,scalar* maximum) {{
      long long node=blockIdx.x;
      __shared__ scalar factors[{splits * heads}], totals[{heads}];
      for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{
        scalar m=-scalar(1.0/0.0);
        for(int s=0;s<{splits};++s) m=fmax(m,maxima[(node*{splits}+s)*{heads}+h]);
        if(m==-scalar(1.0/0.0)) m=0;
        scalar z=0;
        for(int s=0;s<{splits};++s) {{
          long long index=(node*{splits}+s)*{heads}+h;
          scalar a=exp(maxima[index]-m);
          factors[s*{heads}+h]=a;
          z+=a*denominators[index];
        }}
        totals[h]=z+scalar({eps:.17g});
        denominator[node*{heads}+h]=totals[h];
        maximum[node*{heads}+h]=m;
      }}
      __syncthreads();
      for(int i=threadIdx.x;i<{width};i+=blockDim.x) {{
        int h=(i%{channels})/{channels // heads};
        scalar value=0;
        for(int s=0;s<{splits};++s)
          value+=factors[s*{heads}+h]*values[(node*{splits}+s)*{width}+i];
        out[node*{width}+i]=value/totals[h];
      }}
    }}"""


def merge_attention(values, denominators, maxima, channels, eps, outputs):
    """Merge unnormalized online attention tiles inside a CUDA operation.

    Values have shape ``(nodes, splits, ..., channels)``; statistics have
    shape ``(nodes, splits, heads)``. Heads partition the channel dimension.
    Empty tiles use zero values/denominators and negative-infinite maxima.
    Epsilon is added only after merging, preserving weighted-softmax semantics.
    Derivatives are supplied by the enclosing registered operation.
    """
    from .runtime import kernels, runtime

    nodes, splits, heads = maxima.shape
    if not nodes:
        return
    width = values.numel() // (nodes * splits)
    code = merge_source(values.dtype, width, channels, heads, splits, eps)
    compiled = kernels([code], values.device)
    arguments = [value.data_ptr() for value in (values, denominators, maxima, *outputs)]
    runtime().launch(
        [(compiled[code], arguments, nodes, 1, 128, 0)],
        torch.cuda.current_stream(values.device).cuda_stream,
    )


@lru_cache(maxsize=32)
def source(dtype, heads, weight_heads):
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
using T = {scalar};
using int64_t = long long;
extern "C" __global__ void run(const T* scores, const T* weight,
    const int64_t* order, const int64_t* ptr, T* output,
    T* denominators, T* maxima, int64_t num_nodes, double epsilon) {{
    const int64_t task = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (task >= num_nodes * {heads}) return;
    const int head = task % {heads}, lane = threadIdx.x % 32;
    const int64_t node = task / {heads}, begin = ptr[node], end = ptr[node + 1];
    T maximum = -T(1.0 / 0.0);
    for (int64_t k = begin + lane; k < end; k += 32) {{
        maximum = fmax(maximum, scores[order[k] * {heads} + head]);
    }}
    for (int stride = 16; stride; stride >>= 1) {{
        maximum = fmax(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
    }}
    if (maximum == -T(1.0 / 0.0)) {{
        for (int64_t k = begin + lane; k < end; k += 32) {{
            output[order[k] * {heads} + head] = 0;
        }}
        if (lane == 0 && denominators) {{
            denominators[task] = T(epsilon);
            maxima[task] = 0;
        }}
        return;
    }}
    T total = 0;
    for (int64_t k = begin + lane; k < end; k += 32) {{
        const int64_t edge = order[k];
        total += exp(scores[edge * {heads} + head] - maximum)
            * weight[edge * {weight_heads} + {0 if weight_heads == 1 else "head"}];
    }}
    for (int stride = 16; stride; stride >>= 1) {{
        total += __shfl_xor_sync(0xffffffff, total, stride);
    }}
    const T denominator = total + T(epsilon);
    if (lane == 0 && denominators) {{
        denominators[task] = denominator;
        maxima[task] = maximum;
    }}
    for (int64_t k = begin + lane; k < end; k += 32) {{
        const int64_t edge = order[k];
        output[edge * {heads} + head] =
            exp(scores[edge * {heads} + head] - maximum) / denominator;
    }}
}}
"""


def normalized_exp_forward(
    scores, weight, target, num_nodes, eps, denominator=None, maximum=None
):
    """Divide shifted exponentials by weighted sums over incoming edges.

    Optional ``denominator`` and ``maximum`` buffers are filled in place.
    """
    from .runtime import kernels, runtime

    result = torch.empty_like(scores)
    if not num_nodes or not scores.shape[1]:
        return result
    order, ptr = prepare_graph(target, target, 1, num_nodes)
    code = source(scores.dtype, scores.shape[1], weight.shape[1])
    compiled = kernels([code], scores.device)
    epsilon = struct.unpack("Q", struct.pack("d", eps))[0]
    runtime().launch(
        [
            (
                compiled[code],
                [
                    scores.data_ptr(),
                    weight.data_ptr(),
                    order.data_ptr(),
                    ptr.data_ptr(),
                    result.data_ptr(),
                    0 if denominator is None else denominator.data_ptr(),
                    0 if maximum is None else maximum.data_ptr(),
                    num_nodes,
                    epsilon,
                ],
                (num_nodes * scores.shape[1] + 3) // 4,
                1,
                128,
                0,
            )
        ],
        torch.cuda.current_stream(scores.device).cuda_stream,
    )
    return result


@lru_cache(maxsize=32)
def adjoint_source(dtype, heads, weight_heads):
    """Generate a node-wise kernel for score and weight adjoints."""
    scalar = "double" if dtype == torch.float64 else "float"
    columns = 1 << (heads - 1).bit_length()
    reduce_weight = (
        f"for(int step={columns // 2};step;step/=2) "
        f"dw+=__shfl_down_sync(0xffffffffu,dw,step,{columns});"
        if weight_heads == 1
        else ""
    )
    return f"""
using T={scalar};
extern "C" __global__ void run(const T* values,const T* grad,const T* weight,
    const long long* order,const long long* ptr,T* grad_scores,T* grad_weight,
    T* totals,long long nodes) {{
  const long long node=(long long)blockIdx.x*4+threadIdx.x/32;
  if(node>=nodes) return;
  const int lane=threadIdx.x%32, head=lane%{columns};
  const long long begin=ptr[node], end=ptr[node+1];
  T total=0;
  if(head<{heads}) {{
    for(long long k=begin+lane/{columns};k<end;k+={32 // columns}) {{
      const long long index=order[k]*{heads}+head;
      total+=values[index]*grad[index];
    }}
  }}
  for(int step={columns};step<32;step*=2)
    total+=__shfl_xor_sync(0xffffffffu,total,step);
  if(lane<{heads}) totals[node*{heads}+head]=total;
  for(long long k=begin;k<end;k+={32 // columns}) {{
    const long long offset=k+lane/{columns};
    const long long edge=offset<end ? order[offset] : 0;
    T dw=0;
    if(offset<end && head<{heads}) {{
      const long long index=edge*{heads}+head;
      grad_scores[index]=values[index]*(grad[index]-weight[edge*{weight_heads}+{0 if weight_heads == 1 else "head"}]*total);
      dw=-values[index]*total;
    }}
    {reduce_weight}
    if(offset<end && head<{weight_heads}) grad_weight[edge*{weight_heads}+head]=dw;
  }}
}}
"""


def normalized_exp(
    scores: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
    eps: float,
) -> torch.Tensor:
    return normalized_exp_forward(scores, weight, target, num_nodes, eps)


def evaluate_adjoint(
    values: torch.Tensor,
    grad: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .runtime import kernels, runtime

    if not 1 <= values.shape[1] <= 32:
        raise ValueError("The fused attention adjoint supports 1 to 32 heads.")
    outputs = attention_adjoint_outputs(values, grad, weight, target, num_nodes)
    if not num_nodes:
        return outputs
    values, grad, weight = (x.contiguous() for x in (values, grad, weight))
    order, ptr = prepare_graph(target, target, 1, num_nodes)
    code = adjoint_source(values.dtype, values.shape[1], weight.shape[1])
    kernel = kernels([code], values.device)[code]
    runtime().launch(
        [
            (
                kernel,
                [x.data_ptr() for x in (values, grad, weight, order, ptr, *outputs)]
                + [num_nodes],
                (num_nodes + 3) // 4,
                1,
                128,
                0,
            )
        ],
        torch.cuda.current_stream(values.device).cuda_stream,
    )
    return outputs
