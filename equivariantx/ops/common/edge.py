"""Differentiable CUDA execution of static edge expressions."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ...utils.autograd import gradient_mask, nested_forward_ad, torch_gradient_mask
from ...utils.metadata import parse_metadata
from ...utils.outputs import edge_outputs
from ...utils.program import next_adjoint, next_tangent
from .. import config
from ..precision import validate_uniform


@torch.library.custom_op("eqx::edge_program", mutates_args=(), device_types="cuda")
def evaluate_cuda(
    metadata: str,
    inputs: list[torch.Tensor],
    source: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    from .. import get_kernel

    return get_kernel("edge_program", "cuda")(
        metadata, inputs, source, target, num_nodes
    )


evaluate_cuda.register_fake(edge_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, values, source, target, ctx.num_nodes = inputs
    ctx.save_for_backward(source, target, *values)
    ctx.set_materialize_grads(False)


def backward(ctx, gradients):
    source, target, *inputs = ctx.saved_tensors
    slots = sorted({slot for _, slot, _ in parse_metadata(ctx.kernel_metadata)[1]})
    required = gradient_mask(inputs, ctx.needs_input_grad[1], offset=0, trailing=2)
    active = tuple(i for i, need in enumerate(required) if need)
    seeds = tuple(slot for slot, grad in zip(slots, gradients) if grad is not None)
    metadata = next_adjoint(ctx.kernel_metadata, active, seeds, len(inputs))
    adjoints = parse_metadata(metadata)[1]
    results = [None] * len(inputs)
    if adjoints:
        values = inputs + [grad for grad in gradients if grad is not None]
        grads = evaluate(metadata, values, source, target, ctx.num_nodes)
        for slot, grad in zip(sorted({x[1] for x in adjoints}), grads):
            results[slot] = grad.reshape(inputs[slot].shape)
    return None, results, None, None, None


evaluate_cuda.register_autograd(backward, setup_context=setup_context)


@torch.compiler.allow_in_graph
class EdgeProgram(torch.autograd.Function):
    """Save operands for recursive adjoints and analytic functional transforms."""

    @staticmethod
    def forward(backend, metadata, num_nodes, source, target, *inputs):
        if backend == "cuda":
            return tuple(
                torch.ops.eqx.edge_program(
                    metadata, list(inputs), source, target, num_nodes
                )
            )
        from .torch.edge import evaluate_torch

        outputs = evaluate_torch(
            metadata, inputs, source, target, num_nodes, optimized=True
        )
        return tuple(value.view_as(value) for value in outputs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.backend, ctx.kernel_metadata, ctx.num_nodes, source, target, *values = (
            inputs
        )
        ctx.save_for_backward(source, target, *values)
        ctx.save_for_forward(source, target, *values)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *cotangents):
        source, target, *inputs = ctx.saved_tensors
        slots = sorted({slot for _, slot, _ in parse_metadata(ctx.kernel_metadata)[1]})
        required = torch_gradient_mask(ctx, ctx.needs_input_grad[5:])
        active = tuple(i for i, needed in enumerate(required) if needed)
        seeds = tuple(
            slot for slot, value in zip(slots, cotangents) if value is not None
        )
        metadata = next_adjoint(ctx.kernel_metadata, active, seeds, len(inputs))
        outputs = parse_metadata(metadata)[1]
        gradients = [None] * len(inputs)
        if outputs:
            values = inputs + [value for value in cotangents if value is not None]
            result = edge_program(
                ctx.backend, metadata, ctx.num_nodes, source, target, *values
            )
            for slot, value in zip(sorted({slot for _, slot, _ in outputs}), result):
                gradients[slot] = value.reshape(inputs[slot].shape)
        gradients = [
            torch.zeros_like(value) if needed and gradient is None else gradient
            for value, needed, gradient in zip(inputs, required, gradients)
        ]
        return None, None, None, None, None, *gradients

    @staticmethod
    def jvp(
        ctx,
        backend_tangent,
        metadata_tangent,
        count_tangent,
        source_tangent,
        target_tangent,
        *tangents,
    ):
        from .torch.edge import evaluate_torch

        source, target, *inputs = ctx.saved_tensors
        directions = [
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(inputs, tangents)
        ]
        metadata = next_tangent(ctx.kernel_metadata, len(inputs))
        return tuple(
            evaluate_torch(
                metadata,
                inputs + directions,
                source,
                target,
                ctx.num_nodes,
                optimized=True,
            )
        )

    @staticmethod
    def vmap(info, in_dims, backend, metadata, num_nodes, source, target, *inputs):
        from .torch.edge import evaluate_torch

        def function(source, target, *values):
            return tuple(
                evaluate_torch(
                    metadata, values, source, target, num_nodes, optimized=True
                )
            )

        outputs = torch.vmap(function, in_dims=in_dims[3:], randomness=info.randomness)(
            source, target, *inputs
        )
        return outputs, (0,) * len(outputs)


def evaluate(metadata, inputs, source, target, num_nodes):
    """Evaluate native edge expressions with functional derivative rules."""
    return list(edge_program("cuda", metadata, num_nodes, source, target, *inputs))


@torch.compiler.allow_in_graph
def edge_program(backend, metadata, num_nodes, source, target, *inputs):
    """Expose compiled programs and nested JVPs while retaining eager CUDA."""
    validate_uniform(inputs)
    if (
        backend == "cuda"
        and config.EDGE_PROGRAM_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        backend = "torch"
    if nested_forward_ad():
        from .torch.edge import evaluate_torch

        return tuple(
            evaluate_torch(metadata, inputs, source, target, num_nodes, optimized=True)
        )
    return EdgeProgram.apply(backend, metadata, num_nodes, source, target, *inputs)
