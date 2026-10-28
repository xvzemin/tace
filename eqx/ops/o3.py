"""O3 operations."""

import torch

from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from .contraction import adjoint_program


@torch.library.custom_op("eqx::o3_contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    program: str,
    source: torch.Tensor | None,
    target: torch.Tensor | None,
    operands: list[torch.Tensor],
    *,
    batch_paths: bool = False,
) -> list[torch.Tensor]:
    """Run sparse contractions outside the tensor tracing boundary."""
    from ..backends import get_kernel

    return get_kernel("o3_contraction", "cuda")(
        metadata, program, source, target, operands, batch_paths=batch_paths
    )


@contraction.register_fake
def contraction_fake(metadata, program, source, target, operands, *, batch_paths=False):
    program = parse_metadata(program)
    results = [None] * (1 + max(slot for _, _, pairs in program for _, slot in pairs))
    for mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


def contraction_setup_context(ctx, inputs, keyword_only_inputs, output):
    metadata, program, source, target, operands = inputs
    ctx.kernel_metadata = metadata
    ctx.program = parse_metadata(program)
    ctx.batch_paths = keyword_only_inputs["batch_paths"]
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def contraction_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    metadata = parse_metadata(ctx.kernel_metadata)
    geometric = len(metadata) > 3
    required = gradient_mask(operands, ctx.needs_input_grad[4])
    needs_grad = list(required)
    if geometric:
        for mapping, _, _ in ctx.program:
            needs_grad[mapping[5]] = False
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        needs_grad,
        metadata[2],
    )
    if geometric:
        terms = {}
        cotangents = {id(value): index for index, value in enumerate(values)}
        for mapping, weighted, pairs in ctx.program:
            index = mapping[5]
            if not required[index]:
                continue
            for output, slot in pairs:
                if grad_outputs[slot] is None or (
                    output == 2 and not operands[mapping[2]].numel()
                ):
                    continue
                destination = destinations.setdefault(index, len(destinations))
                replacement = list(mapping)
                replacement[output] = cotangents[id(grad_outputs[slot])]
                replacement.append(index)
                key = tuple(replacement), metadata[2] and (weighted or output in (1, 2))
                terms.setdefault(key, []).append((len(mapping), destination))
        program += tuple(
            (mapping, weighted, tuple(pairs))
            for (mapping, weighted), pairs in terms.items()
        )
    gradients = [None] * len(operands)
    if program:
        results = contraction(
            ctx.kernel_metadata,
            repr(program),
            source,
            target,
            values,
            batch_paths=ctx.batch_paths,
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, None, gradients


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)
