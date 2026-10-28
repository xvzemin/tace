"""Channel-wise local products without expanded edge intermediates."""

import torch

from ..utils.metadata import parse_metadata


def local_product(metadata, *features):
    """Contract packed edge features using sparse channel-wise paths."""
    unique, mapping = [], []
    for feature in features:
        index = next((i for i, value in enumerate(unique) if value is feature), None)
        if index is None:
            index = len(unique)
            unique.append(feature)
        mapping.append(index)
    operands = [x.flatten(1) for x in unique]
    mapping.append(len(operands))
    operands.append(features[0].new_empty((0, operands[0].shape[-1])))
    program = repr(((tuple(mapping), len(features), 0, 0),))
    return channel_product(metadata, program, operands)[0]


@torch.library.custom_op(
    "eqx::local_channel_product", mutates_args=(), device_types="cuda"
)
def channel_product(
    metadata: str, program: str, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate sparse diagonal products and their recursive transposes."""
    from ..backends import get_kernel

    return get_kernel("local_channel_product", "cuda")(metadata, program, operands)


@channel_product.register_fake
def channel_product_fake(metadata, program, operands):
    width, dims, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (max(slot for _, _, slot, _ in terms) + 1)
    for mapping, role, slot, _ in terms:
        if outputs[slot] is None:
            rows = operands[mapping[role]].size(0)
            outputs[slot] = operands[0].new_empty(
                (1 if rows == 1 else operands[0].shape[0], dims[role] * width)
            )
    return outputs


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, operands = inputs
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*operands)


def backward(ctx, grad_outputs):
    operands = list(ctx.saved_tensors)
    values = list(operands)
    cotangents = {}
    for slot, grad in enumerate(grad_outputs):
        if grad is not None:
            cotangents[slot] = len(values)
            values.append(grad)
    terms, destinations = [], {}
    for mapping, output, slot, required in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and ctx.needs_input_grad[2][index]:
                destination = destinations.setdefault(index, len(destinations))
                # A transpose must not reintroduce monomials removed by an
                # earlier derivative with respect to an absent operand.
                terms.append(
                    (tuple(replacement), role, destination, required | (1 << output))
                )
    gradients = [None] * len(operands)
    if terms:
        results = channel_product(ctx.kernel_metadata, repr(tuple(terms)), values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, gradients


channel_product.register_autograd(backward, setup_context=setup_context)
