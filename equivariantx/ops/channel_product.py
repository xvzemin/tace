"""Channel-wise local products without expanded edge intermediates."""

import math

import torch

from ..utils.metadata import parse_metadata
from ..utils.outputs import channel_product_outputs


def project(metadata, features):
    """Apply the native Cartesian projector and its recursive transpose."""
    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        raise ValueError("The CUDA projector requires float32 or float64 CUDA tensors.")
    dim = features.shape[-1]
    operands = [
        features.reshape(math.prod(features.shape[:-1]), dim),
        features.new_empty((0, dim)),
    ]
    return channel_product(metadata, repr((((0, 1), 1, 0, 0),)), operands)[0].reshape(
        features.shape
    )


def polynomial_contraction(metadata, inputs, weight, shape, output_dim):
    """Evaluate all paths using CUDA and recursive multilinear adjoints."""
    values = (*inputs, weight)
    if any(
        not x.is_cuda or x.dtype not in (torch.float32, torch.float64) for x in values
    ):
        raise ValueError(
            "The CUDA contraction requires float32 or float64 CUDA tensors."
        )
    if any(x.device != inputs[0].device or x.dtype != inputs[0].dtype for x in values):
        raise ValueError("Inputs and weights must have the same device and dtype.")
    rows = math.prod(shape)
    operands = [
        x.reshape(1, x.shape[-1])
        if i and math.prod(x.shape[:-1]) == 1
        else x.expand(*shape, x.shape[-1]).reshape(rows, x.shape[-1])
        for i, x in enumerate(values)
    ]
    operands.append(inputs[0].new_empty((0, output_dim)))
    role = len(values)
    program = repr(((tuple(range(role + 1)), role, 0, 0),))
    return channel_product(metadata, program, operands)[0].reshape(*shape, output_dim)


def tensor_product(metadata, x, y, weight, output_dim):
    """Evaluate supported CUDA paths without a PyTorch fallback."""
    if metadata is None:
        raise NotImplementedError(
            "The CUDA tensor product supports uuu and u1u paths only."
        )
    if not all(value.is_cuda for value in (x, y, weight)):
        raise ValueError("backend='cuda' requires CUDA inputs and weights.")
    if (
        x.dtype not in (torch.float32, torch.float64)
        or not x.dtype == y.dtype == weight.dtype
    ):
        raise ValueError("CUDA inputs and weights must share float32 or float64 dtype.")
    if not x.device == y.device == weight.device:
        raise ValueError("CUDA inputs and weights must share a device.")
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
    rows = math.prod(shape)
    operands = [
        value.reshape(1, value.shape[-1])
        if i and math.prod(value.shape[:-1]) == 1
        else value.expand(*shape, value.shape[-1]).reshape(rows, value.shape[-1])
        for i, value in enumerate((x, y, weight))
    ]
    operands.append(x.new_empty((0, output_dim)))
    result = channel_product(metadata, repr((((0, 1, 2, 3), 3, 0, 0),)), operands)[0]
    return result.reshape(*shape, output_dim)


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


channel_product.register_fake(channel_product_outputs)


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
