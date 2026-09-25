"""Segmented attention normalization with optional nonnegative edge weights."""

import struct
from functools import lru_cache

import torch

from ..kernels.recompute import Replay
from .graph import prepare_graph


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
    """Merge unnormalized online attention tiles inside a native operation.

    Values have shape ``(nodes, splits, ..., channels)``; statistics have
    shape ``(nodes, splits, heads)``. Heads partition the channel dimension.
    Empty tiles use zero values/denominators and negative-infinite maxima.
    Epsilon is added only after merging, preserving weighted-softmax semantics.
    Derivatives are supplied by the enclosing registered operation.
    """
    from ..kernels.cuda import kernels, runtime

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


def graph_softmax(scores, target, num_nodes, weight=None, eps=1e-16, *, fused=True):
    """Normalize ``(edges, heads)`` scores over incoming edges of each node.

    ``weight`` multiplies the shifted exponential before normalization and may
    have shape ``(edges, 1)`` or ``(edges, heads)``. Maxima are computed from
    scores alone and detached. Epsilon is added to the shifted denominator.
    Empty neighborhoods require no special values. Derivatives include edge
    weights, including weights equal to zero, and support higher orders.
    Set ``fused=False`` to retain the pure PyTorch path on CUDA as well.
    """
    if weight is None:
        weight = scores.new_ones((scores.shape[0], 1))
    if scores.ndim != 2 or weight.ndim != 2:
        raise ValueError("Scores and weights must have shape (edges, heads).")
    if weight.shape[0] != scores.shape[0] or weight.shape[1] not in (
        1,
        scores.shape[1],
    ):
        raise ValueError("Edge weights must broadcast over score heads.")
    if weight.device != scores.device or weight.dtype != scores.dtype:
        raise ValueError("Scores and edge weights must have the same device and dtype.")
    if (
        target.ndim != 1
        or target.shape[0] != scores.shape[0]
        or target.device != scores.device
    ):
        raise ValueError("Target indices must match the edges and device of scores.")
    if fused and scores.is_cuda and scores.dtype in (torch.float32, torch.float64):
        return (
            normalized_exp(
                scores.contiguous(), weight.contiguous(), target.long(), num_nodes, eps
            )
            * weight
        )
    maxima = scores.new_full((num_nodes, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(
        0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
    )
    maxima = torch.where(torch.isneginf(maxima), 0, maxima)
    exp = (scores - maxima[target]).exp()
    denominator = scores.new_zeros((num_nodes, scores.shape[1])).index_add(
        0, target, exp * weight
    )
    return exp * weight / (denominator[target] + eps)


@lru_cache(maxsize=32)
def source(dtype, heads, weight_heads):
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
    using scalar={scalar};
    extern "C" __global__ void run(const scalar* scores, const scalar* weight,
        const long long* order, const long long* ptr, scalar* out, double epsilon) {{
        int h=blockIdx.y, t=threadIdx.x;
        long long begin=ptr[blockIdx.x], end=ptr[blockIdx.x+1];
        if(begin==end) return;
        __shared__ scalar values[128];
        scalar maximum=-scalar(1.0/0.0);
        for(long long k=begin+t;k<end;k+=128)
            maximum=fmax(maximum,scores[order[k]*{heads}+h]);
        values[t]=maximum; __syncthreads();
        for(int stride=64;stride;stride>>=1) {{
            if(t<stride) values[t]=fmax(values[t],values[t+stride]);
            __syncthreads();
        }}
        maximum=values[0]; __syncthreads();
        if(maximum==-scalar(1.0/0.0)) {{
            for(long long k=begin+t;k<end;k+=128) out[order[k]*{heads}+h]=0;
            return;
        }}
        scalar total=0;
        for(long long k=begin+t;k<end;k+=128) {{
            long long edge=order[k];
            total+=exp(scores[edge*{heads}+h]-maximum)*weight[edge*{weight_heads}+{0 if weight_heads == 1 else "h"}];
        }}
        values[t]=total; __syncthreads();
        for(int stride=64;stride;stride>>=1) {{
            if(t<stride) values[t]+=values[t+stride];
            __syncthreads();
        }}
        scalar denominator=values[0]+scalar(epsilon);
        for(long long k=begin+t;k<end;k+=128) {{
            long long edge=order[k];
            out[edge*{heads}+h]=exp(scores[edge*{heads}+h]-maximum)/denominator;
        }}
    }}"""


@torch.library.custom_op(
    "eqx::graph_normalized_exp", mutates_args=(), device_types="cuda"
)
def normalized_exp(
    scores: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
    eps: float,
) -> torch.Tensor:
    from ..kernels.cuda import kernels, runtime

    result = torch.empty_like(scores)
    if scores.numel():
        order = prepare_graph(target, target, 1)
        counts = target.new_zeros(num_nodes).index_add(
            0, target, torch.ones_like(target)
        )
        ptr = torch.cat((target.new_zeros(1), counts.cumsum(0)))
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
                        epsilon,
                    ],
                    ptr.numel() - 1,
                    scores.shape[1],
                    128,
                    0,
                )
            ],
            torch.cuda.current_stream(scores.device).cuda_stream,
        )
    return result


@normalized_exp.register_fake
def normalized_exp_fake(scores, weight, target, num_nodes, eps):
    return torch.empty_like(scores)


def setup_context(ctx, inputs, output):
    scores, weight, target, ctx.num_nodes, eps = inputs
    ctx.save_for_backward(output, weight, target)


def backward(ctx, grad):
    exp, weight, target = ctx.saved_tensors
    reduction = grad.new_zeros((ctx.num_nodes, grad.shape[1])).index_add(
        0, target, grad * exp
    )
    reduction = reduction[target]
    grad_scores = exp * (grad - weight * reduction)
    grad_weight = (-exp * reduction).sum_to_size(weight.shape)
    return grad_scores, grad_weight, None, None, None


normalized_exp.register_autograd(backward, setup_context=setup_context)


class StreamingGraphAttention(Replay):
    """Stream a score/value function over graph edges with online softmax.

    Parameters
    ----------
    function : callable
        Takes a list of tile inputs and returns scores ``(edges, heads)``
        and values ``(edges, angular, channels)``.
    fake_function : callable
        Returns empty node-output, denominator and maximum tensors from input
        shapes. The first input determines the number of nodes.
    edge_axes : dict of int to int
        Maps edge-dependent input slots to their edge axis. Other inputs are
        shared across tiles.
    target_slot : int
        Slot holding receiver indices, or a two-row edge index.
    normalizer_slot, value_weight_slot : int
        Slots holding nonnegative softmax weights and multiplicative value
        weights. They may identify the same tensor.
    valid_slot : int
        Slot holding a boolean mask for padded edges.
    tile_size : int
        Maximum number of edges evaluated together.
    eps : float
        Added to the denominator after shifting by the receiver maximum.

    Notes
    -----
    Only node outputs, shifted denominators and detached maxima are saved.
    The adjoint recomputes each score/value tile once. Its tensor expression
    is itself replayable, including derivatives through the saved outputs
    and denominators, so force training and higher derivatives are supported.
    The callback must be deterministic and state-free. All differentiable
    tensors used by it must be supplied as inputs, rather than captured values.
    """

    def __init__(
        self,
        function,
        fake_function,
        edge_axes,
        target_slot,
        normalizer_slot,
        value_weight_slot,
        valid_slot,
        tile_size,
        eps,
    ):
        self.edge_function = function
        self.edge_axes = edge_axes
        self.target_slot = target_slot
        self.normalizer_slot = normalizer_slot
        self.value_weight_slot = value_weight_slot
        self.valid_slot = valid_slot
        self.tile_size = tile_size
        self.eps = eps
        super().__init__(self.forward, fake_function=fake_function)
        self.saved_outputs = (0, 1, 2)
        self.nondifferentiable = (2,)

    def tiles(self, inputs):
        edges = inputs[self.valid_slot].shape[0]
        for start in range(0, max(edges, 1), self.tile_size):
            count = min(self.tile_size, edges - start)
            yield [
                value.narrow(self.edge_axes[i], start, count)
                if i in self.edge_axes
                else value
                for i, value in enumerate(inputs)
            ]

    def forward(self, inputs, create_graph):
        result, denominator, maximum = self.fake_function(inputs)
        result.zero_()
        denominator.zero_()
        maximum.fill_(-torch.inf)
        nodes, angular, channels = result.shape
        heads = denominator.shape[1]
        for values in self.tiles(inputs):
            index = values[self.target_slot]
            target = index[1] if index.ndim == 2 else index
            scores, value = self.edge_function(values)
            scores = torch.where(values[self.valid_slot][:, None], scores, -torch.inf)
            local_max = torch.full_like(maximum, -torch.inf).scatter_reduce(
                0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
            )
            next_max = torch.maximum(maximum, local_max)
            shift = torch.where(torch.isneginf(next_max), 0, next_max)
            scale = (maximum - shift).exp()
            exp = (scores - shift[target]).exp()
            weight = exp * values[self.normalizer_slot]
            weighted_value = value.view(-1, angular, heads, channels // heads)
            weighted_value = (
                weighted_value
                * (weight * values[self.value_weight_slot])[:, None, :, None]
            )
            result = (
                result.view(nodes, angular, heads, channels // heads)
                * scale[:, None, :, None]
            ).reshape_as(result)
            result = result.index_add(0, target, weighted_value.reshape_as(value))
            denominator = (denominator * scale).index_add(0, target, weight)
            maximum = next_max
        denominator = denominator + self.eps
        result = (
            result.view(nodes, angular, heads, channels // heads)
            / denominator[:, None, :, None]
        ).reshape_as(result)
        return result, denominator, torch.where(torch.isneginf(maximum), 0, maximum)

    def transpose(self, active, outputs, num_inputs):
        key = active, outputs, num_inputs
        if key not in self.derivatives:

            def tile_adjoint(values, create_graph):
                arguments = values[:num_inputs]
                result, denominator, maximum, grad_result, grad_denominator = values[
                    num_inputs:
                ]
                nodes, angular, channels = result.shape
                heads = denominator.shape[1]
                index = arguments[self.target_slot]
                target = index[1] if index.ndim == 2 else index
                scores, value = self.edge_function(arguments)
                scores = torch.where(
                    arguments[self.valid_slot][:, None], scores, -torch.inf
                )
                normalizer = arguments[self.normalizer_slot]
                value_weight = arguments[self.value_weight_slot]
                exp = (scores - maximum[target]).exp()
                quotient = exp / denominator[target]
                incoming = grad_result[target].view(
                    -1, angular, heads, channels // heads
                )
                dot_value = (value.view_as(incoming) * incoming).sum((1, 3))
                dot_result = (
                    (result * grad_result)
                    .view(nodes, angular, heads, channels // heads)
                    .sum((1, 3))[target]
                )
                grad_normalizer = (
                    quotient * (value_weight * dot_value - dot_result)
                    + exp * grad_denominator[target]
                )
                grad_value_weight = quotient * normalizer * dot_value
                grad_scores = grad_normalizer * normalizer
                grad_value = (
                    incoming * (quotient * normalizer * value_weight)[:, None, :, None]
                ).reshape_as(value)
                pairs = [
                    (x, g)
                    for x, g in (
                        (scores, grad_scores),
                        (value, grad_value),
                        (normalizer, grad_normalizer.sum_to_size(normalizer.shape)),
                        (
                            value_weight,
                            grad_value_weight.sum_to_size(value_weight.shape),
                        ),
                    )
                    if x.requires_grad
                ]
                if pairs:
                    result_values, vectors = zip(*pairs)
                    gradients = torch.autograd.grad(
                        result_values,
                        [arguments[i] for i in active],
                        vectors,
                        create_graph=create_graph,
                        allow_unused=True,
                    )
                else:
                    gradients = (None,) * len(active)
                return tuple(
                    torch.zeros_like(arguments[i]) if g is None else g
                    for i, g in zip(active, gradients)
                )

            tile_program = Replay(tile_adjoint, active, capture=False)
            tile_program.resource = self.resource

            def adjoint(values, create_graph):
                arguments = values[:num_inputs]
                result, denominator, maximum = values[num_inputs : num_inputs + 3]
                vectors = dict(zip(outputs, values[num_inputs + 3 :]))
                grad_result = vectors.get(0, torch.zeros_like(result))
                grad_denominator = vectors.get(1, torch.zeros_like(denominator))
                shared = [None] * len(active)
                edges = [[] for _ in active]
                for chunk in self.tiles(arguments):
                    gradients = tile_program(
                        *chunk,
                        result,
                        denominator,
                        maximum,
                        grad_result,
                        grad_denominator,
                    )
                    for slot, (i, grad) in enumerate(zip(active, gradients)):
                        if i in self.edge_axes:
                            edges[slot].append(grad)
                        else:
                            shared[slot] = (
                                grad if shared[slot] is None else shared[slot] + grad
                            )
                return tuple(
                    torch.cat(edges[slot], dim=self.edge_axes[i])
                    if i in self.edge_axes
                    else shared[slot]
                    for slot, i in enumerate(active)
                )

            self.derivatives[key] = Replay(adjoint, active)
            self.derivatives[key].resource = self.resource
        return self.derivatives[key]
