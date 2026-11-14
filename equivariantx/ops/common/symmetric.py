"""Symmetric polynomials and recursive multilinear adjoints."""

import math
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement

import torch

from ...utils.autograd import gradient_mask
from ...utils.metadata import parse_metadata
from ...utils.outputs import symmetric_outputs
from .polynomial import native_polynomial, polynomial_adjoint


def rational_sqrt(numerator, denominator):
    """Evaluate a rational times a square root after exact integer reduction."""
    outside, inside, prime = 1, 1, 2
    while prime * prime <= numerator:
        exponent = 0
        while numerator % prime == 0:
            numerator //= prime
            exponent += 1
        outside *= prime ** (exponent // 2)
        inside *= prime ** (exponent % 2)
        prime += 1
    ratio = Fraction(outside, denominator)
    return ratio.numerator / ratio.denominator * math.sqrt(inside * numerator)


@lru_cache(maxsize=16)
@torch.compiler.assume_constant_result
def matrix_metadata(rank):
    """Describe the normalized symmetric-power action of a 3-by-3 matrix."""
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("The symmetric rank must be a non-negative integer.")
    basis = [(rank - k, k - z, z) for k in range(rank + 1) for z in range(k + 1)]
    lookup = {power: i for i, power in enumerate(basis)}
    rows = []
    for factors in combinations_with_replacement(range(9), rank):
        counts = tuple(factors.count(i) for i in range(9))
        output = tuple(sum(counts[3 * i + j] for j in range(3)) for i in range(3))
        source = tuple(sum(counts[3 * i + j] for i in range(3)) for j in range(3))
        numerator = math.prod(math.factorial(i) for i in (*source, *output))
        denominator = math.prod(math.factorial(i) for i in counts)
        destination = lookup[source] * len(basis) + lookup[output]
        rows.append((factors, -1, destination, rational_sqrt(numerator, denominator)))
    return repr((9, 0, len(basis) ** 2, rank, tuple(rows)))


def matrix_inputs(matrix, rank):
    """Use one scalar channel per matrix without materializing tensor powers."""
    dim = (rank + 1) * (rank + 2) // 2
    values = matrix.reshape(math.prod(matrix.shape[:-2]), 9, 1)
    types = torch.zeros(values.shape[0], device=matrix.device, dtype=torch.int64)
    operands = (
        values,
        matrix.new_empty((1, 0, 1)),
        matrix.new_empty((0, dim * dim, 1)),
    )
    mapping = (*([0] * rank), 1, 2)
    return repr(((mapping, rank + 1, 0, 0),)), types, operands


def symmetric_inputs(features, weight, correlation, output_dim):
    """Describe repeated feature slots without materializing tensor powers."""
    value = features.transpose(1, 2)
    mapping = (*([0] * correlation), 1, 2)
    placeholder = value.new_empty((0, output_dim, features.shape[1]))
    return repr(((mapping, correlation + 1, 0, 0),)), [value, weight, placeholder]


def symmetric_result(output, layout, output_dims):
    """Restore the declared angular/channel layout after contraction."""
    if layout == "ir_mul" or not output_dims:
        return output.flatten(1)
    return torch.cat(
        [part.transpose(1, 2).flatten(1) for part in output.split(output_dims, 1)], 1
    )


@torch.library.custom_op(
    "eqx::symmetric_contract", mutates_args=(), device_types="cuda"
)
def symmetric_contract_cuda(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate element-weighted symmetric contractions."""
    from .. import get_kernel

    return get_kernel("symmetric_contract", "cuda")(
        metadata, program, node_type, operands
    )


symmetric_contract_cuda.register_fake(symmetric_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, node_type, operands = inputs
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_outputs):
    node_type, *operands = ctx.saved_tensors
    needed = gradient_mask(operands, ctx.needs_input_grad[3], offset=1)
    program, values, destinations = polynomial_adjoint(
        repr(ctx.program), operands, grad_outputs, needed
    )
    gradients = [None] * len(operands)
    if destinations:
        results = symmetric_contract(ctx.kernel_metadata, program, node_type, values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, gradients


symmetric_contract_cuda.register_autograd(backward, setup_context=setup_context)


def symmetric_contract(metadata, program, node_type, operands):
    """Evaluate native polynomials with explicit JVP and batching rules."""
    return list(native_polynomial("symmetric", metadata, program, node_type, *operands))


from ..decompositions import register_symmetric_decomposition

register_symmetric_decomposition()
