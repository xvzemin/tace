"""Native CUDA local updates with receiver-normalized rotary attention."""

from functools import lru_cache

import torch

from .program import build, decode, first_adjoint, next_adjoint


def stream(
    module,
    features,
    radial,
    projection,
    bias,
    edge_index,
    cutoff,
    wigner,
    wigner_inv,
    radial_basis,
):
    """Fuse the complete edge update without graph replay or Python callbacks.

    Receiver-wise tiles compute scores and messages together, accumulating
    the shifted denominator and weighted output online. Independent tiles
    are merged without evaluating edges again. No full-edge convolution
    weights or local features are allocated.
    """
    if bias is None:
        bias = projection.new_zeros(projection.shape[1])
    if cutoff is None:
        cutoff = radial.new_ones((radial.shape[0], 1))
    if not features.is_cuda or module._eqx_metadata is None:
        return module(
            features,
            radial @ projection + bias,
            edge_index,
            cutoff,
            wigner,
            wigner_inv,
            radial_basis,
        )
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("The fused local interaction requires float32 or float64.")
    result = interaction(
        module._eqx_metadata,
        edge_index[0],
        edge_index[1],
        [
            module.reshape_in(features),
            radial,
            projection,
            bias,
            cutoff,
            wigner,
            wigner_inv,
            radial_basis,
            *module.parameters(),
        ],
    )[0]
    return module.reshape_out.inverse(result)


def base_program(metadata, inputs):
    return build(
        metadata,
        inputs[1].shape[1],
        inputs[7].shape[1],
        inputs[5].shape[2],
        inputs[6].shape[1],
    )


@lru_cache(maxsize=64)
def forward_program(base):
    nodes, score, value, *_ = base
    return repr((nodes, ((score, 0, "target"), (value, 1, "target"))))


@torch.library.custom_op("eqx::tece_interaction", mutates_args=(), device_types="cuda")
def interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    from ...codegen import launch

    inputs = [x.contiguous() for x in inputs]
    source, target = source.contiguous(), target.contiguous()
    base = base_program(metadata, inputs)
    for value, (width, kind) in zip(inputs, base[3]):
        rows = (
            inputs[0].shape[0]
            if kind in ("source", "target")
            else source.numel()
            if kind == "edge"
            else 1
        )
        if value.numel() != rows * width:
            raise ValueError("Input shape does not match the local interaction layout.")
    result, denominator, maximum = interaction_fake(metadata, source, target, inputs)
    if not source.numel():
        result.zero_()
        denominator.fill_(base[-1])
        maximum.zero_()
        return [result, denominator, maximum]
    launch(
        forward_program(base),
        inputs,
        source,
        target,
        [result, denominator, maximum],
        mode="online",
        heads=base[4],
        channels=base[5],
        eps=base[-1],
    )
    return [result, denominator, maximum]


@interaction.register_fake
def interaction_fake(metadata, source, target, inputs):
    channels, heads = decode(metadata)[2:4]
    nodes = inputs[0].shape[0]
    return [
        inputs[0].new_empty((nodes, inputs[6].shape[1], channels)),
        inputs[0].new_empty((nodes, heads)),
        inputs[0].new_empty((nodes, heads)),
    ]


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, source, target, values = inputs
    ctx.save_for_backward(source, target, *values, *output)
    ctx.num_inputs = len(values)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def backward(ctx, gradients):
    source, target, *saved = ctx.saved_tensors
    values = saved[: ctx.num_inputs]
    result, denominator, maximum = saved[ctx.num_inputs :]
    grad_result, grad_denominator, _ = gradients
    grad_result = torch.zeros_like(result) if grad_result is None else grad_result
    grad_denominator = (
        torch.zeros_like(denominator) if grad_denominator is None else grad_denominator
    )
    active = tuple(i for i, need in enumerate(ctx.needs_input_grad[3]) if need)
    output = [None] * ctx.num_inputs
    if active:
        metadata = first_adjoint(base_program(ctx.kernel_metadata, values), active)
        inputs = [*values, result, denominator, maximum, grad_result, grad_denominator]
        slots = tuple(sorted({slot for _, slot, _ in decode(metadata)[1]}))
        for slot, value in zip(slots, contraction(metadata, source, target, inputs)):
            output[slot] = value
    return None, None, None, output


interaction.register_autograd(backward, setup_context=setup_context)


@torch.library.custom_op("eqx::tece_contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate recursively differentiated local expressions as native kernels."""
    from ...codegen import launch

    inputs = [x.contiguous() for x in inputs]
    result = contraction_fake(metadata, source, target, inputs)
    for value in result:
        value.zero_()
    if result and source.numel():
        launch(metadata, inputs, source.contiguous(), target.contiguous(), result)
    return result


@contraction.register_fake
def contraction_fake(metadata, source, target, inputs):
    slots = sorted({slot for _, slot, _ in decode(metadata)[1]})
    return [
        torch.empty_like(inputs[slot], memory_format=torch.contiguous_format)
        for slot in slots
    ]


def contraction_setup_context(ctx, inputs, output):
    ctx.kernel_metadata, source, target, values = inputs
    ctx.save_for_backward(source, target, *values)
    ctx.set_materialize_grads(False)


def contraction_backward(ctx, gradients):
    source, target, *inputs = ctx.saved_tensors
    slots = sorted({slot for _, slot, _ in decode(ctx.kernel_metadata)[1]})
    seeds = tuple(slot for slot, grad in zip(slots, gradients) if grad is not None)
    active = tuple(i for i, need in enumerate(ctx.needs_input_grad[3]) if need)
    result = [None] * len(inputs)
    if active and seeds:
        metadata = next_adjoint(ctx.kernel_metadata, active, seeds, len(inputs))
        slots_out = sorted({slot for _, slot, _ in decode(metadata)[1]})
        values = contraction(
            metadata,
            source,
            target,
            inputs + [grad for grad in gradients if grad is not None],
        )
        for slot, value in zip(slots_out, values):
            result[slot] = value
    return None, None, None, result


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)
