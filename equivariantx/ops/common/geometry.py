"""Direction derivatives of aligned contractions in the rotation Lie algebra."""

import torch

from ...utils.autograd import gradient_mask
from ...utils.metadata import kernel_plan, parse_metadata
from ...utils.outputs import direction_outputs


@torch.library.custom_op("eqx::direction_contraction", mutates_args=())
def direction_contraction(
    metadata: str,
    program: str,
    vectors: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
    use_cuda: bool,
) -> list[torch.Tensor]:
    """Evaluate angular contractions without independent matrix adjoints."""
    from .. import get_kernel

    backend = "cuda" if vectors.is_cuda and use_cuda else "torch"
    return get_kernel("direction_contraction", backend)(
        metadata, program, vectors, source, target, operands, use_cuda
    )


@direction_contraction.register_fake
def direction_fake(metadata, program, vectors, source, target, operands, use_cuda):
    return direction_outputs(program, operands)


def setup_context(ctx, inputs, output):
    metadata, program, vectors, source, target, operands, use_cuda = inputs
    ctx.kernel_metadata = metadata
    ctx.use_cuda = use_cuda
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(vectors, source, target, *operands)


def backward(ctx, grad_outputs):
    vectors, source, target, *operands = ctx.saved_tensors
    required = (ctx.needs_input_grad[2], False, False, *ctx.needs_input_grad[5])
    if ctx.use_cuda:
        required = gradient_mask(
            (vectors, source, target, *operands), required, offset=0
        )
    values = list(operands)
    cotangents = {}
    for slot, gradient in enumerate(grad_outputs):
        if gradient is not None:
            cotangents[slot] = len(values)
            values.append(gradient)
    terms, destinations = {}, {}
    direction_slot = None
    placeholder = len(values)
    if required[0]:
        values.append(vectors.new_empty(1).expand_as(vectors))
    plan = kernel_plan(ctx.kernel_metadata)
    for rank, mapping, weighted_only, pairs in ctx.program:
        for output, slot in pairs:
            if slot not in cotangents or (
                output == 2 and not operands[mapping[2]].numel()
            ):
                continue
            replacement = list(mapping)
            replacement[output] = cotangents[slot]
            weighted = plan.has_unweighted and (weighted_only or output in (1, 2))
            key = rank, tuple(replacement), weighted
            for role, index in enumerate(mapping):
                if role not in (output, 3, 4) and required[3 + index]:
                    destination = destinations.setdefault(index, len(destinations))
                    terms.setdefault(key, []).append((role, destination))
    if required[0]:
        direction_slot = len(destinations)
        for rank, mapping, weighted_only, pairs in ctx.program:
            for output, slot in pairs:
                if slot not in cotangents or (
                    output == 2 and not operands[mapping[2]].numel()
                ):
                    continue
                replacement = list(mapping)
                replacement[output] = cotangents[slot]
                replacement.append(placeholder)
                weighted = plan.has_unweighted and (weighted_only or output in (1, 2))
                key = rank + 1, tuple(replacement), weighted
                terms.setdefault(key, []).append((7 + rank, direction_slot))
    gradients = [None] * len(operands)
    direction_gradient = None
    if terms:
        program = tuple((*key, tuple(pairs)) for key, pairs in terms.items())
        results = direction_contraction(
            ctx.kernel_metadata,
            repr(program),
            vectors,
            source,
            target,
            values,
            ctx.use_cuda,
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
        if direction_slot is not None:
            # d n = omega x n. Radial amplitudes remain separate operands.
            if vectors.is_cuda and ctx.use_cuda:
                from .wigner import alignment_cuda

                direction_gradient = alignment_cuda(
                    repr("direction_gradient"), [vectors, results[direction_slot]]
                )[0]
            else:
                direction_gradient = torch.linalg.cross(
                    results[direction_slot], vectors
                ) / vectors.square().sum(-1, keepdim=True)
    return None, None, direction_gradient, None, None, gradients, None


direction_contraction.register_autograd(backward, setup_context=setup_context)
