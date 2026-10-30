"""Segmented attention normalization with optional nonnegative edge weights."""

import torch

from ..ops.attention import normalized_exp
from ..ops.recompute import Replay


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

    Examples
    --------
    >>> from eqx.conv import graph_softmax
    >>> scores = torch.zeros(3, 1)
    >>> target = torch.tensor([0, 0, 1])
    >>> graph_softmax(scores, target, num_nodes=3, fused=False)
    tensor([[0.5000],
            [0.5000],
            [1.0000]])
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
