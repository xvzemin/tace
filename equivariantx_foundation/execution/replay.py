"""CUDA Graph execution for stateless tensor programs."""

from collections.abc import Sequence
from threading import get_ident

import torch


def replay_capacity(value):
    """Return a single atom/edge capacity or a sorted tuple of fixed capacities."""

    def sequence(x):
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes))

    def valid(x):
        return (
            sequence(x)
            and len(x) in (2, 3)
            and all(type(n) is int for n in x)
            and x[0] > 0
            and x[1] >= 0
            and (len(x) == 2 or x[2] > 0)
        )

    if valid(value):
        return tuple(value)
    if sequence(value) and len(value) and all(valid(v) and len(v) == 3 for v in value):
        return tuple(
            sorted({tuple(v) for v in value}, key=lambda v: (v[0] + v[1], v[0], v[2]))
        )
    raise ValueError(
        "replay_bucket requires [max_atoms, max_edges, optional max_graphs] or a "
        "nonempty list of triples, with positive atoms and graphs and nonnegative edges."
    )


class ReplayContext:
    """Thread, stream and precision settings of a CUDA runner."""

    def __init__(self, device):
        self.stream = torch.cuda.current_stream(device)
        self.device = self.stream.device
        self.thread = get_ident()
        self.tf32 = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        self.check()

    def check(self):
        if (
            get_ident() != self.thread
            or torch.cuda.current_stream(self.device) != self.stream
        ):
            raise ValueError(
                "Use the thread and CUDA stream on which replay was created."
            )
        if torch.is_autocast_enabled("cuda") or self.tf32 != (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ):
            raise ValueError(
                "Replay requires unchanged TF32 settings and no CUDA autocast."
            )


class TensorProgramReplay:
    """Replay a deterministic CUDA tensor program at one input signature.

    Parameters
    ----------
    runner : callable
        Capture-safe tensor program. AOTI runners must be single-threaded.
    inputs : tuple of torch.Tensor
        CUDA input tensors with fixed shapes and strides.
    context : ReplayContext, optional
        Existing execution context for the runner.
    """

    @staticmethod
    def input_device(inputs):
        """Return the common CUDA device of the flat input tensors."""
        if not isinstance(inputs, (tuple, list)) or not inputs:
            raise ValueError("Replay requires a nonempty tuple of CUDA tensors.")
        if not all(
            isinstance(x, torch.Tensor) and x.is_cuda and x.layout == torch.strided
            for x in inputs
        ):
            raise ValueError("Replay requires strided CUDA tensor inputs.")
        device = inputs[0].device
        if any(x.device != device for x in inputs):
            raise ValueError("Replay inputs must all be on the same CUDA device.")
        return device

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, runner, inputs, *, context=None):
        from equivariantx.ops.common.cuda.graph import retain_constants

        self.runner = runner
        self.device = self.input_device(inputs)
        self.context = ReplayContext(self.device) if context is None else context
        if self.context.device != self.device:
            raise ValueError("Replay context and input devices differ.")
        self.context.check()
        self.inputs = tuple(x.clone() for x in inputs)
        if any(a.stride() != b.stride() for a, b in zip(self.inputs, inputs)):
            raise ValueError("Replay requires dense, non-overlapping input layouts.")
        self.constants = {}
        with torch.cuda.device(self.device), retain_constants(self.constants):
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(self.context.stream)
            with torch.cuda.stream(side):
                runner(*self.inputs)
            side.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.graph, stream=side, capture_error_mode="thread_local"
            ):
                self.outputs = runner(*self.inputs)
        self.output_tree = None
        if isinstance(self.outputs, torch.Tensor):
            leaves = (self.outputs,)
        elif type(self.outputs) in (tuple, list) and all(
            isinstance(x, torch.Tensor) for x in self.outputs
        ):
            leaves = self.outputs
        else:
            from torch.utils._pytree import tree_flatten

            self.outputs, self.output_tree = tree_flatten(self.outputs)
            leaves = self.outputs
        if not all(
            x is None or (isinstance(x, torch.Tensor) and x.device == self.device)
            for x in leaves
        ):
            raise ValueError("Replay output leaves must be CUDA tensors or None.")

    @torch.no_grad()
    def __call__(self, *inputs):
        self.context.check()
        if len(inputs) != len(self.inputs) or any(
            not isinstance(x, torch.Tensor)
            or x.layout != torch.strided
            or x.device != buf.device
            or x.dtype != buf.dtype
            or x.shape != buf.shape
            or x.stride() != buf.stride()
            for buf, x in zip(self.inputs, inputs)
        ):
            raise ValueError("Replay input shape, stride, dtype or device changed.")
        with torch.cuda.device(self.device):
            for buf, value in zip(self.inputs, inputs):
                buf.copy_(value)
            self.graph.replay()
            if self.output_tree is not None:
                from torch.utils._pytree import tree_unflatten

                return tree_unflatten(
                    [None if x is None else x.clone() for x in self.outputs],
                    self.output_tree,
                )
            if isinstance(self.outputs, torch.Tensor):
                return self.outputs.clone()
            return type(self.outputs)(x.clone() for x in self.outputs)
