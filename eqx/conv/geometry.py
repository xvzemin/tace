"""Direction derivatives of aligned contractions in the rotation Lie algebra."""

import math
from functools import lru_cache
from string import ascii_letters

import torch
from e3nn import o3

from .convolution import kernel_plan, parse_program


@lru_cache(maxsize=32)
def generators(degree):
    """Return real rotation generators in the harmonic basis, on the CPU."""
    if not degree:
        return torch.zeros(3, 1, 1, dtype=torch.float64, device="cpu")
    return -math.sqrt(degree * (degree + 1) * (2 * degree + 1)) * o3.wigner_3j(
        degree, 1, degree, dtype=torch.float64, device="cpu"
    ).permute(1, 2, 0)


@lru_cache(maxsize=256)
def angular_coefficients(metadata, rank):
    """Differentiate fixed angular tensors, including their vector indices.

    A rank-zero tensor is the order-zero CG slice. Each further index is a
    rotation derivative. Differentiating every angular index also accounts
    for the moving local frame in higher derivatives.
    """
    plan = kernel_plan(metadata)
    coefficients = []
    previous = angular_coefficients(metadata, rank - 1) if rank else None
    for index, ((_, path), entries) in enumerate(
        zip(plan.path_data, plan.sparse_paths)
    ):
        dim, dim_out = path[4:6]
        if not rank:
            value = torch.zeros(dim, dim_out, dtype=torch.float64, device="cpu")
            for a, b, coefficient in entries:
                value[a, b] = coefficient
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
                        base, -generators(degree), dims=([axis], [1])
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
    for result in results:
        result.zero_()
    plan = kernel_plan(metadata)
    if source.numel() and plan.path_data:
        calls = [
            (
                rank,
                tuple(role for role, _ in pairs),
                tuple(operands[i] for i in mapping),
                tuple(results[slot] for _, slot in pairs),
                weighted,
            )
            for rank, mapping, weighted, pairs in parse_program(program)
        ]
        if vectors.is_cuda and use_cuda:
            from .cuda import contract_directions

            if vectors.dtype not in (torch.float32, torch.float64):
                raise TypeError("The CUDA convolution requires float32 or float64.")
            contract_directions(metadata, source, target, calls)
        else:
            for rank, outputs, values, destinations, weighted in calls:
                for output, result in zip(outputs, destinations):
                    reference(
                        metadata, rank, output, source, target, values, result, weighted
                    )
    return results


@direction_contraction.register_fake
def direction_fake(metadata, program, vectors, source, target, operands, use_cuda):
    program = parse_program(program)
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
    ctx.program = parse_program(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(vectors, source, target, *operands)


def backward(ctx, grad_outputs):
    vectors, source, target, *operands = ctx.saved_tensors
    values = list(operands)
    cotangents = {}
    for slot, gradient in enumerate(grad_outputs):
        if gradient is not None:
            cotangents[slot] = len(values)
            values.append(gradient)
    terms, destinations = {}, {}
    direction_slot = None
    placeholder = len(values)
    if ctx.needs_input_grad[2]:
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
                if role not in (output, 3, 4) and ctx.needs_input_grad[5][index]:
                    destination = destinations.setdefault(index, len(destinations))
                    terms.setdefault(key, []).append((role, destination))
    if ctx.needs_input_grad[2]:
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


def reference(metadata, rank, output, source, target, operands, result, weighted_only):
    """Reference contraction for directions and arbitrary derivative order."""
    plan = kernel_plan(metadata)
    x, radial, projection, din, dout, amplitudes, y, *vectors = operands
    edges = source.numel()
    projected = projection.numel() != 0
    labels = "".join(letter for letter in ascii_letters if letter not in "aebkmnuv")
    if rank > len(labels):
        raise ValueError("The reference einsum exhausted its angular index labels.")
    for index, (mode, path) in enumerate(plan.path_data):
        start, end, mul, mul_out, dim, dim_out, dstart, dend, weight, harmonic = path
        if ((weighted_only or output in (1, 2)) and weight < 0) or (
            output == 2 and not projected
        ):
            continue
        if rank and index in plan.scalar_paths:
            continue
        cg = angular_coefficients(metadata, rank)[index].to(x)
        if index in plan.scalar_paths:
            di = torch.eye(dim, dtype=x.dtype, device=x.device).expand(edges, -1, -1)
            do = di
        else:
            di = (
                din[:, dstart : dstart + dim * dim]
                .reshape(-1, dim, dim)
                .expand(edges, -1, -1)
            )
            do = (
                dout[:, dend : dend + dim_out * dim_out]
                .reshape(-1, dim_out, dim_out)
                .expand(edges, -1, -1)
            )
        tensors = [
            None
            if output == 0
            else x[source, start : start + mul * dim].reshape(edges, dim, mul),
            None,
            None,
            di,
            do,
            None if output == 5 else amplitudes[:, harmonic].expand(edges),
            None
            if output == 6
            else y[target, end : end + mul_out * dim_out].reshape(
                edges, dim_out, mul_out
            ),
        ]
        tokens = ["eau", "ek", "kuv", "ema", "enb", "e", "ebv"]
        width = mul if mode == "uvu" else mul * mul_out
        weight_shape = (mul,) if mode == "uvu" else (mul, mul_out)
        if mode == "uvu":
            tokens[2], tokens[6] = "ku", "ebu"
        if weight >= 0:
            if projected:
                if output != 1:
                    tensors[1] = radial.expand(edges, -1)
                if output != 2:
                    tensors[2] = projection[:, weight : weight + width].reshape(
                        projection.size(0), *weight_shape
                    )
            else:
                tokens[1] = "eu" if mode == "uvu" else "euv"
                if output != 1:
                    tensors[1] = (
                        radial[:, weight : weight + width]
                        .reshape(-1, *weight_shape)
                        .expand(edges, *weight_shape)
                    )
        if rank:
            rotation = din[:, 1:10].reshape(-1, 3, 3).expand(edges, -1, -1)
            for axis, vector in enumerate(vectors):
                tokens.append("e" + labels[axis])
                tensors.append(
                    None
                    if output == 7 + axis
                    else torch.einsum("eab,eb->ea", rotation, vector.expand(edges, -1))
                )
        active = [
            (token, value) for token, value in zip(tokens, tensors) if value is not None
        ]
        value = torch.einsum(
            ",".join([token for token, _ in active] + ["mn" + labels[:rank]])
            + "->"
            + tokens[output],
            *[value for _, value in active],
            cg,
        )
        if output in (0, 6):
            nodes = source if output == 0 else target
            begin, size = (
                (start, mul * dim) if output == 0 else (end, mul_out * dim_out)
            )
            result[:, begin : begin + size].index_add_(
                0, nodes, value.reshape(edges, size)
            )
        elif output >= 7:
            value = torch.einsum("eab,ea->eb", rotation, value)
            result.add_(value if result.size(0) != 1 else value.sum(0, keepdim=True))
        elif output == 5:
            result[:, harmonic].add_(
                value if result.size(0) != 1 else value.sum().reshape(1)
            )
        elif output == 2:
            result[:, weight : weight + width].add_(
                value.reshape(projection.size(0), width)
            )
        elif projected:
            result.add_(value if result.size(0) != 1 else value.sum(0, keepdim=True))
        else:
            value = value.reshape(edges, width)
            result[:, weight : weight + width].add_(
                value if result.size(0) != 1 else value.sum(0, keepdim=True)
            )
