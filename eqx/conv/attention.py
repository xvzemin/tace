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
    """Merge unnormalized online attention tiles inside a CUDA operation.

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
    """Normalize attention scores over incoming edges of each node.

    Parameters
    ----------
    scores : torch.Tensor
        Attention logits with shape ``(edges, heads)``.
    target : torch.Tensor
        Destination node indices with shape ``(edges,)``.
    num_nodes : int
        Number of destination nodes, including isolated nodes.
    weight : torch.Tensor, optional
        Nonnegative edge weights with shape ``(edges, 1)`` or
        ``(edges, heads)``. Defaults to one.
    eps : float, optional
        Constant added to the shifted denominator. Defaults to ``1e-16``.
    fused : bool, optional
        Use the CUDA kernel for supported inputs. Set ``False`` to use
        PyTorch on either device. Defaults to ``True``.

    Returns
    -------
    torch.Tensor
        Weighted softmax values with shape ``(edges, heads)``.

    Notes
    -----
    Maxima are detached. Derivatives through scores and weights support
    higher orders, including weights equal to zero.
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
    from ..kernels.cuda import kernels, runtime

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
    return normalized_exp_forward(scores, weight, target, num_nodes, eps)


@normalized_exp.register_fake
def normalized_exp_fake(scores, weight, target, num_nodes, eps):
    return torch.empty_like(scores)


def setup_context(ctx, inputs, output):
    scores, weight, target, ctx.num_nodes, eps = inputs
    ctx.save_for_backward(output, weight, target)


@lru_cache(maxsize=32)
def adjoint_program(heads, weight_heads):
    """Build the grouped dot product and normalized-exponential adjoints."""
    from .program import Program

    program = Program()
    exp = program.input(0, "edge", heads)
    grad = program.input(1, "edge", heads)
    dot = program.binary("mul", exp, grad)
    reduction = repr((tuple(program.nodes), ((dot, 0, "target"),)))

    weight = program.input(2, "edge", weight_heads)
    total = program.input(3, "target", heads)
    broadcast = program.gather(weight, (0,) * heads) if weight_heads == 1 else weight
    grad_scores = program.binary(
        "mul",
        exp,
        program.binary(
            "add", grad, program.scale(program.binary("mul", broadcast, total), -1)
        ),
    )
    grad_weight = program.scale(program.binary("mul", exp, total), -1)
    if weight_heads == 1 and heads != 1:
        grad_weight = program.scatter(grad_weight, (0,) * heads, 1)
    pointwise = repr(
        (
            tuple(program.nodes),
            (
                (grad_scores, 0, "edge"),
                (grad_weight, 1, "edge"),
            ),
        )
    )
    return reduction, pointwise


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


@torch.library.custom_op(
    "eqx::graph_normalized_exp_adjoint", mutates_args=(), device_types="cuda"
)
def evaluate_adjoint(
    values: torch.Tensor,
    grad: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ..kernels.cuda import kernels, runtime

    if not 1 <= values.shape[1] <= 32:
        raise ValueError("The fused attention adjoint supports 1 to 32 heads.")
    outputs = adjoint_fake(values, grad, weight, target, num_nodes)
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


@evaluate_adjoint.register_fake
def adjoint_fake(values, grad, weight, target, num_nodes):
    return (
        torch.empty_like(values, memory_format=torch.contiguous_format),
        torch.empty_like(weight, memory_format=torch.contiguous_format),
        values.new_empty((num_nodes, values.shape[1])),
    )


def adjoint_setup_context(ctx, inputs, output):
    values, grad, weight, target, ctx.num_nodes = inputs
    ctx.save_for_backward(values, grad, weight, target, output[2])
    ctx.set_materialize_grads(False)


def adjoint_backward(ctx, grad_scores, grad_weight, grad_total):
    from ..utils.metadata import parse_metadata
    from .edge import evaluate
    from .program import next_adjoint

    values, grad, weight, target, total = ctx.saved_tensors
    reduction, pointwise = adjoint_program(int(values.shape[1]), int(weight.shape[1]))
    seeds = tuple(i for i, x in enumerate((grad_scores, grad_weight)) if x is not None)
    active = tuple(i for i in range(3) if ctx.needs_input_grad[i])
    dot_active = tuple(i for i in active if i < 2)
    results = [None] * 3
    if seeds:
        metadata = next_adjoint(
            pointwise, active + ((3,) if dot_active else ()), seeds, 4
        )
        operands = [values, grad, weight, total]
        operands.extend(x for x in (grad_scores, grad_weight) if x is not None)
        adjoints = evaluate(metadata, operands, target, target, ctx.num_nodes)
        slots = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
        for slot, value in zip(slots, adjoints):
            if slot == 3:
                grad_total = value if grad_total is None else grad_total + value
            else:
                results[slot] = value
    if grad_total is not None and dot_active:
        metadata = next_adjoint(reduction, dot_active, (0,), 2)
        dot_results = evaluate(
            metadata, [values, grad, grad_total], target, target, ctx.num_nodes
        )
        for i, value in zip(dot_active, dot_results):
            results[i] = value if results[i] is None else results[i] + value
    return *results, None, None


evaluate_adjoint.register_autograd(
    adjoint_backward, setup_context=adjoint_setup_context
)


def backward(ctx, grad):
    exp, weight, target = ctx.saved_tensors
    # Keep small reductions and compiler-fused pointwise operations native.
    if (
        not torch.compiler.is_compiling()
        and 8 <= exp.shape[1] <= 32
        and exp.numel() >= 8_000_000
    ):
        grad_scores, grad_weight, _ = evaluate_adjoint(
            exp, grad, weight, target, ctx.num_nodes
        )
        return grad_scores, grad_weight, None, None, None
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
        and values ``(edges, angular, channels)``. Must be deterministic and
        stateless, with all differentiable tensors passed as inputs.
    fake_function : callable
        Returns empty node-output, denominator and maximum tensors from input
        shapes. The first input determines the number of nodes.
    edge_axes : dict of int to int
        Maps edge-dependent input slots to their edge axis. Other inputs are
        shared across tiles.
    target_slot : int
        Slot holding target indices, or a two-row edge index.
    normalizer_slot, value_weight_slot : int
        Slots holding nonnegative softmax weights and multiplicative value
        weights. They may identify the same tensor.
    valid_slot : int
        Slot holding a boolean mask for padded edges.
    tile_size : int
        Maximum number of edges evaluated together.
    eps : float
        Added to the denominator after shifting by the target-node maximum.
    backend : {"cuda", "torch"}, optional
        ``"cuda"`` replays tiles during differentiation. ``"torch"`` retains
        the PyTorch online-softmax graph and uses automatic differentiation.
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
        *,
        backend="cuda",
    ):
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
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

    def __call__(self, *inputs):
        if self.backend == "torch":
            return self.forward(inputs, create_graph=torch.is_grad_enabled())
        return super().__call__(*inputs)

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
