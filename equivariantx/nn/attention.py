"""Streaming graph attention."""

import torch

from ..ops import config, resolve_backend
from ..ops.common.recompute import Replay
from ..ops.precision import validate_precision
from ..ops.torch.nn import attention as torch_kernel


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
    >>> from equivariantx.nn import StreamingGraphAttention
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
        if tile_size < 1:
            raise ValueError("tile_size must be positive.")
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
        if edges <= self.tile_size:
            yield inputs
            return
        for start in range(0, edges, self.tile_size):
            count = min(self.tile_size, edges - start)
            yield [
                value.narrow(self.edge_axes[i], start, count)
                if i in self.edge_axes
                else value
                for i, value in enumerate(inputs)
            ]

    def forward(self, inputs, create_graph):
        validate_precision(inputs)
        if self.backend == "torch" and config.TORCH_REFERENCE:
            return torch_kernel.reference(self, inputs)
        return torch_kernel.stream(self, inputs, create_graph)

    def transpose(self, active, outputs, num_inputs):
        from ..ops.cuda.nn import attention as cuda_kernel

        return cuda_kernel.transpose(self, active, outputs, num_inputs)
