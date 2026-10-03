"""Native CUDA local updates with receiver-normalized rotary attention."""

import torch

from ....conv.contraction import gradient_mask
from ....conv.program import next_adjoint
from ....utils.metadata import parse_metadata
from .program import build, first_adjoint


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
    *,
    radial_network=None,
):
    """Fuse the complete edge update without graph replay or Python callbacks.

    Edge tiles use matrix products for dense contractions and CUDA kernels
    for intervening expressions. Only attention scores span all edges;
    convolution weights and local features are recomputed in bounded tiles.
    """
    if bias is None:
        bias = projection.new_zeros(projection.shape[1])
    if cutoff is None:
        cutoff = features.new_ones((edge_index.shape[1], 1))
    if not features.is_cuda or module._eqx_metadata is None:
        if radial_network is not None:
            from ....conv.network import materialize

            radial = radial_network(materialize(radial, edge_index))
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
    inputs = [
        module.reshape_in(features),
        radial,
        projection,
        bias,
        cutoff,
        wigner,
        wigner_inv,
        radial_basis,
        *module.parameters(),
    ]
    metadata = module._eqx_metadata
    if radial_network is not None:
        from ....conv.network import compose_radial

        nodes, score, value, specs, *settings = build(
            metadata,
            projection.shape[0],
            radial_basis.shape[1],
            wigner.shape[2],
            wigner_inv.shape[1],
        )
        program, inputs = compose_radial(
            repr((nodes, ((score, 0, "edge"), (value, 1, "edge")))),
            inputs,
            1,
            radial_network,
        )
        nodes, outputs = parse_metadata(program)
        specs = list(specs) + [None] * (len(inputs) - len(specs))
        for op, width, _, data in nodes:
            if op == "input":
                specs[data[0]] = (width, data[1])
        base = (nodes, outputs[0][0], outputs[1][0], tuple(specs), *settings)
        metadata = repr((metadata, base))
    result = interaction(
        metadata,
        edge_index[0],
        edge_index[1],
        inputs,
    )[0]
    return module.reshape_out.inverse(result)


def base_program(metadata, inputs):
    description = parse_metadata(metadata)
    if len(description) == 2:
        return description[1]
    return build(
        metadata,
        inputs[1].shape[1],
        inputs[7].shape[1],
        inputs[5].shape[2],
        inputs[6].shape[1],
    )


@torch.library.custom_op("eqx::tece_interaction", mutates_args=(), device_types="cuda")
def interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    from .execution import forward

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
    forward(base, inputs, source, target, [result, denominator, maximum])
    return [result, denominator, maximum]


@interaction.register_fake
def interaction_fake(metadata, source, target, inputs):
    description = parse_metadata(metadata)
    if len(description) == 2:
        description = parse_metadata(description[0])
    channels, heads = description[2:4]
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
    required = gradient_mask(values, ctx.needs_input_grad[3])
    active = tuple(i for i, need in enumerate(required) if need)
    output = [None] * ctx.num_inputs
    if active:
        metadata = first_adjoint(base_program(ctx.kernel_metadata, values), active)
        inputs = [*values, result, denominator, maximum, grad_result, grad_denominator]
        slots = tuple(sorted({slot for _, slot, _ in parse_metadata(metadata)[1]}))
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
    from ....conv.execution import launch

    inputs = [x.contiguous() for x in inputs]
    result = contraction_fake(metadata, source, target, inputs)
    for value in result:
        value.zero_()
    if result and source.numel():
        launch(metadata, inputs, source.contiguous(), target.contiguous(), result)
    return result


@contraction.register_fake
def contraction_fake(metadata, source, target, inputs):
    slots = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
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
    slots = sorted({slot for _, slot, _ in parse_metadata(ctx.kernel_metadata)[1]})
    seeds = tuple(slot for slot, grad in zip(slots, gradients) if grad is not None)
    required = gradient_mask(inputs, ctx.needs_input_grad[3])
    active = tuple(i for i, need in enumerate(required) if need)
    result = [None] * len(inputs)
    if active and seeds:
        metadata = next_adjoint(ctx.kernel_metadata, active, seeds, len(inputs))
        slots_out = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
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
