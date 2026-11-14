"""O3 operations."""

import torch

from ...utils.autograd import gradient_mask
from ...utils.metadata import parse_metadata
from ...utils.outputs import o3_outputs
from .contraction import adjoint_program, project_radial
from ..precision import validate_uniform


@torch.library.custom_op("eqx::o3_contraction", mutates_args=(), device_types="cuda")
def contraction_cuda(
    metadata: str,
    program: str,
    source: torch.Tensor | None,
    target: torch.Tensor | None,
    operands: list[torch.Tensor],
    *,
    batch_paths: bool = False,
) -> list[torch.Tensor]:
    """Run sparse contractions outside the tensor tracing boundary."""
    from .. import get_kernel

    return get_kernel("o3_contraction", "cuda")(
        metadata, program, source, target, operands, batch_paths=batch_paths
    )


contraction_cuda.register_fake(o3_outputs)


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
    angular = (
        len(metadata) > 5
        and metadata[5]
        and not (isinstance(metadata[3], tuple) and metadata[3][0] == "transverse")
    )
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
    if geometric and not angular:
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
    if angular:
        indices = tuple(
            sorted(
                {mapping[5] for mapping, _, _ in ctx.program if required[mapping[5]]}
            )
        )
        if indices:
            partials = angular_base_gradient(
                ctx.kernel_metadata,
                ctx.program,
                source,
                target,
                operands,
                grad_outputs,
                indices,
            )
            for index, value in zip(indices, partials):
                gradients[index] = (
                    value if gradients[index] is None else gradients[index] + value
                )
    return None, None, None, None, gradients


def angular_base_gradient(
    metadata, program, source, target, operands, cotangents, indices
):
    """Differentiate base coordinates while holding Lie directions fixed.

    Appending a generator computes a rotation derivative, not a Cartesian
    derivative of the base vector. Differentiate the exact Torch polynomial
    for this role; multilinear factor transposes retain their native kernels.
    """
    from .torch.o3 import contraction as torch_contraction

    terms = tuple(
        (
            (*mapping[:5], len(operands) + indices.index(mapping[5]), *mapping[6:])
            if mapping[5] in indices
            else mapping,
            weighted,
            pairs,
        )
        for mapping, weighted, pairs in program
    )

    def function(*vectors):
        return tuple(
            torch_contraction(
                metadata, repr(terms), source, target, [*operands, *vectors]
            )
        )

    outputs, pullback = torch.func.vjp(
        function, *(operands[index] for index in indices)
    )
    return pullback(
        tuple(
            torch.zeros_like(value) if grad is None else grad
            for value, grad in zip(outputs, cotangents)
        )
    )


contraction_cuda.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)


@torch.compiler.allow_in_graph
def contraction(metadata, program, source, target, operands, *, batch_paths=False):
    """Expose supported O3 contractions inside functional transformations."""
    validate_uniform(operands)
    description = parse_metadata(metadata)
    if len(description) > 6 and description[6]:
        offset = description[6]
        values, directions, terms = list(operands), {}, []
        for mapping, weighted, pairs in parse_metadata(program):
            index = mapping[5]
            if index not in directions:
                directions[index] = len(values)
                values.append(operands[index][:, offset : offset + 3])
            mapping = (*mapping[:5], directions[index], *mapping[6:])
            terms.append((mapping, weighted, pairs))
        metadata = repr((*description[:6], 0, *description[7:]))
        program, operands = repr(tuple(terms)), values
    if torch._C._are_functorch_transforms_active():
        from .torch.o3 import contraction as torch_contraction

        return torch_contraction(
            metadata, program, source, target, operands, batch_paths=batch_paths
        )
    program, operands = project_radial(program, operands, 4)
    return torch.ops.eqx.o3_contraction(
        metadata, program, source, target, operands, batch_paths=batch_paths
    )
