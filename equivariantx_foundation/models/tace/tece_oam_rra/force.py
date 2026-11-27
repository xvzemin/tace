"""Indexed rotary edge expansions and frozen force intermediates."""

from functools import lru_cache

import torch

from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.metadata import compact_metadata, parse_metadata
from equivariantx_foundation.ops.tece import (
    contraction,
    tece_contraction_outputs,
    tece_outputs,
)

from .expressions import base_program, first_adjoint

EDGE_TILE = 8192


def primal_program(metadata):
    plan = parse_metadata(metadata)
    if len(plan) == 2 and isinstance(plan[0], int) and isinstance(plan[1], str):
        return plan
    return None, metadata


def capped_program(metadata, limit):
    """Encode the maximum number of retained wide edge rows."""
    return compact_metadata(repr((limit, compact_metadata(metadata))))


@torch.library.custom_op(
    "eqx_tece_rra::indexed_expansion", mutates_args=(), device_types="cuda"
)
def indexed_expansion(
    metadata: str, scales: list[float], hidden: torch.Tensor, coefficient: torch.Tensor
) -> torch.Tensor:
    """Apply the local edge expansion to indexed attention features."""
    from .expansion import expansion

    return expansion(parse_metadata(metadata), tuple(scales), hidden, coefficient)[0]


@indexed_expansion.register_fake
def indexed_output(metadata, scales, hidden, coefficient):
    return hidden.new_empty((hidden.shape[0], 1440))


@torch.library.custom_op(
    "eqx_tece_rra::indexed_expansion_vjp", mutates_args=(), device_types="cuda"
)
def indexed_expansion_vjp(
    metadata: str,
    scales: list[float],
    hidden: torch.Tensor,
    coefficient: torch.Tensor,
    grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the hidden and coefficient derivatives of the local expansion."""
    from .expansion import expansion

    return expansion(parse_metadata(metadata), tuple(scales), hidden, coefficient, grad)


@indexed_expansion_vjp.register_fake
def indexed_input_grads(metadata, scales, hidden, coefficient, grad):
    return hidden.new_empty(hidden.shape), coefficient.new_empty(coefficient.shape)


@torch.library.custom_op(
    "eqx_tece_rra::indexed_expansion_dual", mutates_args=(), device_types="cuda"
)
def indexed_expansion_dual(
    metadata: str,
    scales: list[float],
    hidden: torch.Tensor,
    coefficient: torch.Tensor,
    grad: torch.Tensor,
    vh: torch.Tensor,
    vc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply both input Hessian-vector products and the output Jacobian-vector product."""
    from .expansion import expansion_dual

    return expansion_dual(
        parse_metadata(metadata), tuple(scales), hidden, coefficient, grad, vh, vc
    )


@indexed_expansion_dual.register_fake
def indexed_dual_outputs(metadata, scales, hidden, coefficient, grad, vh, vc):
    return (
        hidden.new_empty(hidden.shape),
        coefficient.new_empty(coefficient.shape),
        grad.new_empty(grad.shape),
    )


def indexed_vjp_context(ctx, inputs, output):
    ctx.metadata_key, ctx.scales, hidden, coefficient, grad = inputs
    ctx.save_for_backward(hidden, coefficient, grad)


def indexed_vjp_backward(ctx, vh, vc):
    h, c, dy = ctx.saved_tensors
    if torch.is_grad_enabled():
        from .expansion import reference_expansion

        def value(x, w):
            return reference_expansion(
                parse_metadata(ctx.metadata_key), ctx.scales, x, w
            )

        def tangent(x, w):
            return torch.func.jvp(value, (x, w), (vh, vc))[1]

        jv, pullback = torch.func.vjp(tangent, h, c)
        result = (*pullback(dy), jv)
    else:
        result = indexed_expansion_dual(ctx.metadata_key, ctx.scales, h, c, dy, vh, vc)
    return None, None, *result


indexed_expansion_vjp.register_autograd(
    indexed_vjp_backward, setup_context=indexed_vjp_context
)


def indexed_context(ctx, inputs, output):
    ctx.kernel_metadata, ctx.scales, hidden, coefficient = inputs
    ctx.save_for_backward(hidden, coefficient)


def indexed_backward(ctx, grad):
    return (
        None,
        None,
        *indexed_expansion_vjp(
            ctx.kernel_metadata, ctx.scales, *ctx.saved_tensors, grad
        ),
    )


indexed_expansion.register_autograd(indexed_backward, setup_context=indexed_context)


@lru_cache(maxsize=16)
def saved_nodes(base):
    nodes, score, *_ = base
    products = [
        args
        for op, _, args, data in nodes
        if op == "product" and data[1] == 5 and len(args) == 6 and args[0] == args[1]
    ]
    if len(products) != 1:
        raise ValueError("Expected one local bilinear rotary-attention product.")
    hidden, _, _, coefficient, _, _ = products[0]
    op, _, operands, _ = nodes[coefficient]
    if op == "mul":
        activations = [i for i in operands if nodes[i][0] == "silu"]
        coefficient = activations[0] if len(activations) == 1 else coefficient
    if nodes[coefficient][0] != "silu":
        raise ValueError("Expected SiLU rotary-attention product coefficients.")
    return score, nodes[coefficient][2][0], hidden


@lru_cache(maxsize=64)
def saved_adjoint(base, active, start, joint=False, *, score_only=False):
    from .expansion import lower_expansion
    from .frame import lower_endpoint

    lowered = lower_expansion(base, saved_nodes(base)) if joint else None
    nodes, outputs = parse_metadata(first_adjoint(lowered or base, active))
    rows = list(nodes)
    ids = saved_nodes(base)[:1] if score_only else saved_nodes(base)
    for slot, i in enumerate(ids, start):
        rows[i] = "input", rows[i][1], (), (slot, "edge")
    nodes = tuple(rows)
    if lowered is not None:
        nodes = lower_endpoint(nodes, outputs)
    return compact_metadata(repr((nodes, outputs)))


@lru_cache(maxsize=16)
def value_plan(base, joint=False):
    from equivariantx.ops.common.cuda.execution import execution_plan

    from .expansion import lower_expansion

    nodes, _, value, *_ = (
        (lower_expansion(base, saved_nodes(base)) or base) if joint else base
    )
    _, coefficient, hidden = saved_nodes(base)
    return execution_plan(
        repr(
            (
                nodes,
                tuple(
                    (i, slot, "edge")
                    for slot, i in enumerate((coefficient, hidden, value))
                ),
            )
        )
    )


@lru_cache(maxsize=16)
def value_tile(base, joint):
    from .expansion import tile_factory

    return tile_factory(value_plan(base, joint)[0])


@lru_cache(maxsize=32)
def adjoint_programs(metadata):
    from .frame import lower_endpoint_matrix

    first, second = parse_metadata(metadata)
    if isinstance(first, str):
        return lower_endpoint_matrix(first), lower_endpoint_matrix(second)
    return lower_endpoint_matrix(metadata), None


@lru_cache(maxsize=32)
def edge_slots(metadata):
    nodes, roots = parse_metadata(metadata)
    inputs = set()
    for op, _, _, spec in nodes:
        if op == "input":
            slot, kind = spec
            if kind == "edge":
                inputs.add(slot)
            elif isinstance(kind, tuple):
                inputs.add(kind[1])
    ordered = sorted({slot for _, slot, _ in roots})
    outputs = {slot for _, slot, kind in roots if kind == "edge"}
    return inputs | outputs, [slot in outputs for slot in ordered]


@torch.library.custom_op(
    "eqx_tece_rra::expansion_pullback", mutates_args=(), device_types="cuda"
)
def expansion_pullback(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate the saved edge-expansion and attention input adjoints."""
    from equivariantx.ops.common.cuda.execution import launch
    from equivariantx_foundation.ops.cuda.tece_attention import BACKWARD_TILE_SIZE

    from .expansion import adjoint_tile

    inputs = [x.contiguous() for x in inputs]
    from .score import lower_pullback

    first, tail = adjoint_programs(metadata)
    if all(t.dtype == torch.float32 for t in inputs):
        first = lower_pullback(first)
        if tail is not None:
            tail = lower_pullback(tail)
    outputs = tece_contraction_outputs(first, source, target, inputs)
    for x in outputs:
        x.zero_()
    if outputs and source.numel():
        source, target = source.contiguous(), target.contiguous()
        rows = source.numel()
        kept = inputs[-1].shape[0] if tail is not None else rows
        if kept == rows:
            launch(
                first,
                inputs,
                source,
                target,
                outputs,
                BACKWARD_TILE_SIZE,
                tile_class=adjoint_tile(first),
            )
        else:
            for begin, end, plan in ((0, kept, first), (kept, rows, tail)):
                if begin == end:
                    continue
                ids, flags = edge_slots(plan)
                section = slice(begin, end)
                launch(
                    plan,
                    [t[section] if i in ids else t for i, t in enumerate(inputs)],
                    source[section],
                    target[section],
                    [t[section] if flag else t for t, flag in zip(outputs, flags)],
                    BACKWARD_TILE_SIZE,
                    tile_class=adjoint_tile(plan),
                )
    return outputs


@expansion_pullback.register_fake
def expansion_outputs(metadata, source, target, inputs):
    return tece_contraction_outputs(
        adjoint_programs(metadata)[0], source, target, inputs
    )


@torch.library.custom_op(
    "eqx_tece_rra::saved_interaction", mutates_args=(), device_types="cuda"
)
def saved_interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate frozen rotary attention and return its force intermediates."""
    from equivariantx.ops.common.cuda.attention import normalized_exp_forward
    from equivariantx.ops.common.cuda.execution import TiledProgram
    from equivariantx.ops.common.cuda.reduction import reduce_messages
    from equivariantx_foundation.ops.cuda.tece_attention import forward_plan

    from .layout import paired_tile
    from .score import lower_primal

    inputs = [x.contiguous() for x in inputs]
    source, target = source.contiguous(), target.contiguous()
    limit, metadata = primal_program(metadata)
    base = base_program(metadata, inputs)
    if all(t.dtype == torch.float32 for t in inputs):
        base = lower_primal(base)
    nodes, score, value, _, _, channels, eps = base
    ids = saved_nodes(base)
    result, denominator, maximum = tece_outputs(metadata, source, target, inputs)
    rows = source.numel()
    capacity = (rows + EDGE_TILE - 1) // EDGE_TILE * EDGE_TILE
    kept = rows if limit is None else min(rows, limit)
    retained = capacity if limit is None else min(capacity, limit)
    saved = [inputs[0].new_empty((capacity, nodes[score][1]))[:rows]]
    saved += [inputs[0].new_empty((retained, nodes[i][1]))[:kept] for i in ids[1:]]
    result.zero_()
    if not rows:
        denominator.fill_(eps)
        maximum.zero_()
        return [result, denominator, maximum, *saved]
    shared = {}
    score_plan = forward_plan(nodes, score)
    score_tile = (
        paired_tile(nodes) if inputs[0].dtype == torch.float32 else TiledProgram
    )
    for begin in range(0, rows, EDGE_TILE):
        end = min(begin + EDGE_TILE, rows)
        tile = score_tile(score_plan, inputs, source, target, begin, end, shared)
        saved[0][begin:end].copy_(tile.value(score))
        del tile
    scale = normalized_exp_forward(
        saved[0], inputs[4], target, result.shape[0], eps, denominator, maximum
    )
    scale.mul_(inputs[4].square())
    joint = inputs[0].dtype == torch.float32
    plan = value_plan(base, joint)
    tile_class = value_tile(base, joint)
    for begin in range(0, rows, EDGE_TILE):
        end = min(begin + EDGE_TILE, rows)
        tile = tile_class(plan, inputs, source, target, begin, end, shared)
        if begin < kept:
            count = min(end, kept) - begin
            for i, tensor in zip(ids[1:], saved[1:]):
                tensor[begin : begin + count].copy_(tile.value(i)[:count])
                tile.release("value", i)
        message = tile.value(value)
        reduce_messages(
            message, tile.target, result.flatten(1), scale[begin:end], channels
        )
        del message, tile
    return [result, denominator, maximum, *saved]


@saved_interaction.register_fake
def saved_outputs(metadata, source, target, inputs):
    limit, metadata = primal_program(metadata)
    base = base_program(metadata, inputs)
    rows = source.shape[0]
    kept = rows if limit is None else torch.sym_min(rows, limit)
    return [
        *tece_outputs(metadata, source, target, inputs),
        *(
            inputs[0].new_empty((rows if j == 0 else kept, base[0][i][1]))
            for j, i in enumerate(saved_nodes(base))
        ),
    ]


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, source, target, values = inputs
    ctx.num_inputs = len(values)
    ctx.save_for_backward(source, target, *values, *output)
    ctx.mark_non_differentiable(*output[2:])
    ctx.set_materialize_grads(False)


@torch.autograd.function.once_differentiable
def backward(ctx, grads):
    source, target, *items = ctx.saved_tensors
    values = items[: ctx.num_inputs]
    result, denominator, maximum, *primals = items[ctx.num_inputs :]
    grad_result, grad_denominator = grads[:2]
    if grad_result is None:
        grad_result = torch.zeros_like(result)
    if grad_denominator is None:
        grad_denominator = torch.zeros_like(denominator)
    active = tuple(
        i for i, on in enumerate(gradient_mask(values, ctx.needs_input_grad[3])) if on
    )
    output = [None] * len(values)
    if active:
        from .expansion import lower_expansion

        limit, description = primal_program(ctx.kernel_metadata)
        base = base_program(description, values)
        args = [*values, result, denominator, maximum, grad_result, grad_denominator]
        joint = (
            values[0].dtype == torch.float32
            and lower_expansion(base, saved_nodes(base)) is not None
        )
        metadata = saved_adjoint(base, active, len(args), joint)
        slots = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
        if limit is not None:
            tail = saved_adjoint(base, active, len(args), joint, score_only=True)
            metadata = compact_metadata(repr((metadata, tail)))
        operator = expansion_pullback if joint or limit is not None else contraction
        for slot, grad in zip(
            slots, operator(metadata, source, target, [*args, *primals])
        ):
            output[slot] = grad
    return None, None, None, output


saved_interaction.register_autograd(backward, setup_context=setup_context)
