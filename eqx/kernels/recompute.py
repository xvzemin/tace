"""Bounded activation replay, including recursively replayed derivatives."""

from collections import OrderedDict
from itertools import count
from threading import RLock
from uuid import uuid4
from weakref import WeakValueDictionary

import torch

_programs = WeakValueDictionary()
_identifiers = count()
_namespace = uuid4().hex


class Replay:
    """Evaluate a tensor function without saving its intermediate activations.

    Derivatives are tensor functions themselves and use the same execution
    path. CUDA graphs amortize launch overhead for repeated tile shapes;
    their input buffers are refreshed on every call, including parameters.
    Returned tensors never alias the reusable graph buffers.
    """

    def __init__(self, function, output_indices=None, fake_function=None, capture=True):
        self.function = function
        self.output_indices = output_indices
        self.fake_function = fake_function
        self.capture = capture
        self.saved_outputs = ()
        self.nondifferentiable = ()
        self.resource = None
        self.key = f"{_namespace}:{next(_identifiers)}"
        self.derivatives = {}
        self.graphs = OrderedDict()
        self.lock = RLock()
        _programs[self.key] = self

    def __call__(self, *inputs):
        return replay(list(inputs), self.key)

    def transpose(self, active, outputs, num_inputs):
        key = active, outputs, num_inputs
        if key not in self.derivatives:

            def derivative(values, create_graph):
                arguments = values[:num_inputs]
                results = self.function(arguments, True)
                pairs = [
                    (results[i], v)
                    for i, v in zip(outputs, values[num_inputs:])
                    if results[i].requires_grad
                ]
                if pairs:
                    result, vector = zip(*pairs)
                    grads = torch.autograd.grad(
                        result,
                        [arguments[i] for i in active],
                        vector,
                        create_graph=create_graph,
                        allow_unused=True,
                    )
                else:
                    grads = (None,) * len(active)
                return tuple(
                    torch.zeros_like(arguments[i]) if g is None else g
                    for i, g in zip(active, grads)
                )

            self.derivatives[key] = Replay(derivative, active, capture=self.capture)
            self.derivatives[key].resource = self.resource
        return self.derivatives[key]

    def evaluate(self, inputs):
        def run(values):
            # Custom-op backends execute below Autograd. Derivative replay
            # builds a temporary graph internally, independently of the
            # registered derivative of the enclosing replay operation.
            with (
                torch._C._SetExcludeDispatchKeyGuard(
                    torch._C.DispatchKey.AutogradFunctionality, False
                ),
                torch.set_grad_enabled(self.output_indices is not None),
            ):
                return tuple(x.detach() for x in self.function(values, False))

        # Small workloads and outer CUDA graph captures use the ordinary
        # execution path. The latter must not start a nested capture.
        if (
            not self.capture
            or not inputs[0].is_cuda
            or inputs[0].numel() < 4096
            or torch.cuda.is_current_stream_capturing()
        ):
            values = tuple(x.detach().requires_grad_(x.requires_grad) for x in inputs)
            return [x.clone() for x in run(values)]

        device = inputs[0].device
        stream = torch.cuda.current_stream(device)
        key = (
            device,
            stream.cuda_stream,
            tuple((x.shape, x.dtype, x.requires_grad) for x in inputs),
        )
        with self.lock, torch.cuda.device(device):
            if key not in self.graphs:
                values = tuple(
                    x.detach().clone().requires_grad_(x.requires_grad) for x in inputs
                )
                capture_stream = torch.cuda.Stream(device=device)
                capture_stream.wait_stream(stream)
                with torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        run(values)
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    result = run(values)
                self.graphs[key] = graph, values, result
                # Dynamic graph sizes must not accumulate unbounded workspaces.
                if len(self.graphs) > 2:
                    self.graphs.popitem(last=False)
            graph, values, result = self.graphs[key]
            self.graphs.move_to_end(key)
            for dst, src in zip(values, inputs):
                dst.detach().copy_(src)
            graph.replay()
            return [x.clone() for x in result]


@torch.library.custom_op("eqx::replay", mutates_args=())
def replay(inputs: list[torch.Tensor], key: str) -> list[torch.Tensor]:
    program = _programs.get(key)
    if program is None:
        raise RuntimeError(
            "Streaming execution plans belong to the live Python model. "
            "Reconstruct the model before evaluating this graph; export a "
            "standalone AOTI package with streaming disabled."
        )
    return program.evaluate(inputs)


@replay.register_fake
def replay_fake(inputs, key):
    program = _programs[key]
    if program.output_indices is not None:
        return [torch.empty_like(inputs[i]) for i in program.output_indices]
    if program.fake_function is not None:
        return list(program.fake_function(inputs))
    return [torch.empty_like(x) for x in program.function(inputs, False)]


def setup_context(ctx, inputs, output):
    values, key = inputs
    ctx.program = _programs[key]
    # A callback may borrow its module weakly to avoid a global cache cycle.
    # Keep that module alive for the lifetime of the derivative graph.
    ctx.resource = None if ctx.program.resource is None else ctx.program.resource()
    ctx.active = tuple(i for i, x in enumerate(values) if x.requires_grad)
    ctx.num_inputs = len(values)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*(output[i] for i in ctx.program.nondifferentiable))
    ctx.save_for_backward(*values, *(output[i] for i in ctx.program.saved_outputs))


def backward(ctx, cotangents):
    inputs = ctx.saved_tensors
    outputs = tuple(i for i, x in enumerate(cotangents) if x is not None)
    gradients = [None] * ctx.num_inputs
    if ctx.active and outputs:
        program = ctx.program.transpose(ctx.active, outputs, ctx.num_inputs)
        result = program(*inputs, *(cotangents[i] for i in outputs))
        for i, grad in zip(ctx.active, result):
            gradients[i] = grad
    return gradients, None


replay.register_autograd(backward, setup_context=setup_context)
