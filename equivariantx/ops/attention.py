"""Graph attention operators and derivatives."""

from functools import lru_cache

import torch


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
    from ..backends import get_kernel

    return get_kernel("graph_normalized_exp", "cuda")(
        scores, weight, target, num_nodes, eps
    )


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
    from ..backends import get_kernel

    return get_kernel("graph_normalized_exp_adjoint", "cuda")(
        values, grad, weight, target, num_nodes
    )


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
