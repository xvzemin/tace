"""Joint expert contractions with recursively transposed derivatives."""

import zlib
from functools import lru_cache

import torch

from ....utils import parse_metadata


def encode_metadata(plans):
    """Keep self-contained kernel plans compact in exported operator schemas."""
    return zlib.compress(repr(tuple(plans)).encode()).hex()


@lru_cache(maxsize=128)
def decode_metadata(metadata):
    return tuple(
        parse_metadata(value)
        for value in parse_metadata(zlib.decompress(bytes.fromhex(metadata)).decode())
    )


@torch.library.custom_op("eqx::bilinear_ace", mutates_args=(), device_types="cuda")
def contract(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    from ....kernels.cuda_graph import execute
    from .bilinear_cuda import launch

    outputs = contract_fake(metadata, program, node_type, operands)
    terms = parse_metadata(program)

    def run(inputs, outputs):
        launch(metadata, terms, inputs[0], inputs[1:], outputs)

    read = {0}
    for _, mapping, output, _ in terms:
        read.update(1 + index for role, index in enumerate(mapping) if role != output)
    execute("ace", (metadata, program), run, (node_type, *operands), outputs, read=read)
    return outputs


@contract.register_fake
def contract_fake(metadata, program, node_type, operands):
    terms = parse_metadata(program)
    outputs = [None] * (max(slot for _, _, _, slot in terms) + 1)
    for plan, mapping, role, slot in terms:
        if outputs[slot] is None:
            rows = operands[mapping[2]].shape[0] if role == 2 else node_type.shape[0]
            outputs[slot] = operands[0].new_empty(
                (rows, decode_metadata(metadata)[plan][0][role])
            )
    return outputs


def setup_context(ctx, inputs, output):
    metadata, program, node_type, operands = inputs
    ctx.plans = metadata
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_outputs):
    node_type, *operands = ctx.saved_tensors
    values = list(operands)
    cotangents = {}
    for slot, grad in enumerate(grad_outputs):
        if grad is not None:
            cotangents[slot] = len(values)
            values.append(grad)
    terms, destinations = [], {}
    for plan, mapping, output, slot in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and ctx.needs_input_grad[3][index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append((plan, tuple(replacement), role, destination))
    gradients = [None] * len(operands)
    if terms:
        results = contract(ctx.plans, repr(tuple(terms)), node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


contract.register_autograd(backward, setup_context=setup_context)
