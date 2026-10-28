"""Direction derivatives of aligned contractions in the rotation Lie algebra."""

from functools import lru_cache

import torch
from e3nn import o3

from ..o3._rotation import so3_generators
from ..utils.autograd import gradient_mask
from ..utils.metadata import parse_metadata
from .o2_o3 import kernel_plan


@lru_cache(maxsize=256)
def harmonic_derivatives(degree, rank):
    """Return angular derivatives at the pole, including rotating vector indices."""
    if rank == 1:
        return so3_generators(degree)[:, :, degree].T.contiguous()
    previous = harmonic_derivatives(degree, rank - 1)
    value = torch.tensordot(so3_generators(degree), previous, dims=([2], [0]))
    value = value.movedim(0, -1).contiguous()
    for axis in range(1, rank):
        term = torch.tensordot(previous, -so3_generators(1), dims=([axis], [1]))
        value += term.movedim(-1, axis)
    value[..., 1] = 0
    return value


@lru_cache(maxsize=256)
def angular_coefficients(metadata, rank):
    """Differentiate fixed angular tensors, including their vector indices.

    A rank-zero tensor is the order-zero CG slice. Derivatives act on the
    harmonic at the pole and its vector indices before coupling to features.
    General order-zero maps without harmonic metadata differentiate every
    angular index instead.
    """
    plan = kernel_plan(metadata)
    coefficients = []
    direct = rank > 0 and bool(plan.harmonic_degrees)
    previous = angular_coefficients(metadata, rank - 1) if rank and not direct else None
    for index, ((_, path), entries) in enumerate(
        zip(plan.path_data, plan.sparse_paths)
    ):
        dim, dim_out = path[4:6]
        if not rank:
            value = torch.zeros(dim, dim_out, dtype=torch.float64, device="cpu")
            for a, b, coefficient in entries:
                value[a, b] = coefficient
        elif direct:
            degree = plan.harmonic_degrees[index]
            cg = o3.wigner_3j(
                (dim - 1) // 2,
                degree,
                (dim_out - 1) // 2,
                dtype=torch.float64,
                device="cpu",
            )
            base = angular_coefficients(metadata, 0)[index]
            scale = (base * cg[:, degree, :]).sum() / cg[:, degree, :].square().sum()
            # Differentiate the harmonic at the pole before coupling it. This
            # preserves its sparse support without subtracting large, nearly
            # cancelling input/output generator contractions.
            harmonic = harmonic_derivatives(degree, rank)
            value = torch.einsum("aqb,q...->ab...", cg, harmonic) * scale
        else:
            value = torch.zeros(
                dim, dim_out, *([3] * rank), dtype=torch.float64, device="cpu"
            )
            if index not in plan.scalar_paths:
                base = previous[index]
                for axis, degree in enumerate(
                    ((dim - 1) // 2, (dim_out - 1) // 2, *([1] * (rank - 1)))
                ):
                    # F(Q n; x) = F(n; D(Q)^T x). The generators are skew.
                    term = torch.tensordot(
                        base, -so3_generators(degree), dims=([axis], [1])
                    )
                    value += term.movedim(-1, axis)
                # Rotation about the aligned direction fixes the direction.
                # Its generator vanishes after summing all angular indices.
                value[..., 1] = 0
        coefficients.append(value)
    return tuple(coefficients)


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
    results = direction_fake(
        metadata, program, vectors, source, target, operands, use_cuda
    )
    plan = kernel_plan(metadata)
    terms = parse_metadata(program)

    def run(inputs, outputs):
        source, target, *values = inputs
        for result in outputs:
            result.zero_()
        if not source.numel() or not plan.path_data:
            return
        calls = [
            (
                rank,
                tuple(role for role, _ in pairs),
                tuple(values[i] for i in mapping),
                tuple(outputs[slot] for _, slot in pairs),
                weighted,
            )
            for rank, mapping, weighted, pairs in terms
        ]
        if vectors.is_cuda and use_cuda:
            from ..backends.cuda.o2_o3.convolution import contract_directions

            if vectors.dtype not in (torch.float32, torch.float64):
                raise TypeError("The CUDA convolution requires float32 or float64.")
            contract_directions(metadata, source, target, calls)
        else:
            from ..backends.torch.geometry import reference

            for rank, outputs, values, destinations, weighted in calls:
                for output, result in zip(outputs, destinations):
                    reference(
                        metadata, rank, output, source, target, values, result, weighted
                    )

    if vectors.is_cuda and use_cuda:
        from ..backends.cuda.graph import convolution

        convolution(
            "o2_o3",
            ("direction", metadata, program),
            run,
            source,
            target,
            operands,
            results,
            tuple((mapping, weighted, pairs) for _, mapping, weighted, pairs in terms),
            6,
        )
    else:
        run((source, target, *operands), results)
    return results


@direction_contraction.register_fake
def direction_fake(metadata, program, vectors, source, target, operands, use_cuda):
    program = parse_metadata(program)
    results = [None] * (
        1 + max(slot for _, _, _, pairs in program for _, slot in pairs)
    )
    for _, mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


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
