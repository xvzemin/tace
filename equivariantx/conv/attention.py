"""Segmented attention normalization with optional nonnegative edge weights."""

import torch

from ..ops import resolve_backend
from ..ops.common.recompute import Replay
from ..ops.precision import validate_precision
from ..ops.torch.conv import attention as torch_kernel


def graph_softmax(
    scores, target, num_nodes, weight=None, eps=1e-16, *, backend=None, fused=None
):
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
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider. CPU inputs use Torch.
    fused : bool, optional
        Compatibility alias: True selects CUDA and False selects Torch.

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
    >>> from equivariantx.conv import graph_softmax
    >>> scores = torch.zeros(3, 1)
    >>> target = torch.tensor([0, 0, 1])
    >>> graph_softmax(scores, target, num_nodes=3)
    tensor([[0.5000],
            [0.5000],
            [1.0000]])
    """
    backend = resolve_backend(backend, scores.device)
    validate_precision((scores, weight))
    if fused is not None:
        backend = "cuda" if fused else "torch"
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
        or target.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Target indices must match the edges and device of scores.")
    target = target.long()
    if backend == "cuda" and scores.is_cuda:
        from ..ops.cuda.conv import attention as cuda_kernel

        return cuda_kernel.graph_softmax(scores, target, num_nodes, weight, eps)
    return torch_kernel.graph_softmax(scores, target, num_nodes, weight, eps)


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
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        ``"cuda"`` replays tiles during differentiation. ``"torch"`` retains
        the PyTorch online-softmax graph and uses automatic differentiation.

    Notes
    -----
    Tracing exposes the tensor program for compiler fusion and standalone
    export. The number of tiles is specialized; compiled activation memory
    can differ from eager CUDA replay.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.conv import StreamingGraphAttention
    >>> def output_shapes(inputs):
    ...     nodes = inputs[0].shape[0]
    ...     return (inputs[2].new_empty(nodes, 1, 2),
    ...             inputs[1].new_empty(nodes, 1), inputs[1].new_empty(nodes, 1))
    >>> layer = StreamingGraphAttention(
    ...     lambda inputs: (inputs[1], inputs[2]), output_shapes,
    ...     edge_axes={i: 0 for i in range(1, 6)}, target_slot=3,
    ...     normalizer_slot=4, value_weight_slot=4, valid_slot=5,
    ...     tile_size=2, eps=1e-8,
    ... )
    >>> inputs = [torch.empty(2, 0), torch.zeros(3, 1), torch.ones(3, 1, 2),
    ...           torch.tensor([0, 1, 0]), torch.ones(3, 1), torch.ones(3, dtype=torch.bool)]
    >>> result, denominator, maximum = layer(*inputs)
    >>> result.shape
    torch.Size([2, 1, 2])
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
        backend=None,
    ):
        backend = resolve_backend(backend)
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
        validate_precision(inputs)
        return torch_kernel.stream(self, inputs, create_graph)

    def transpose(self, active, outputs, num_inputs):
        from ..ops.cuda.conv import attention as cuda_kernel

        return cuda_kernel.transpose(self, active, outputs, num_inputs)
