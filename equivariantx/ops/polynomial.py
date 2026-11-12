"""Symmetric polynomial schedules shared by execution providers."""

from collections import defaultdict
from functools import lru_cache

import torch

from ..utils.autograd import nested_forward_ad, torch_gradient_mask
from ..utils.metadata import parse_metadata
from .precision import validate_uniform


@lru_cache(maxsize=128)
def polynomial_groups(metadata, program):
    """Combine permutation-equivalent products without merging learned paths."""
    _, _, _, degree, rows = parse_metadata(metadata)
    groups = defaultdict(lambda: defaultdict(float))
    for mapping, role, slot, required in parse_metadata(program):
        for indices, weight, output, coefficient in rows:
            indices = (*indices, *([-1] * (degree - len(indices))), weight, output)
            if indices[role] < 0 or any(
                index < 0 for k, index in enumerate(indices) if required & (1 << k)
            ):
                continue
            factors = tuple(
                sorted(
                    (mapping[k], index, k == degree)
                    for k, index in enumerate(indices)
                    if k != role and index >= 0
                )
            )
            groups[slot, role == degree, indices[role]][factors] += coefficient
    return tuple(
        (
            key,
            tuple(
                (factors, value) for factors, value in sorted(terms.items()) if value
            ),
        )
        for key, terms in sorted(groups.items())
    )


@lru_cache(maxsize=256)
def program_terms(kind, program):
    """Normalize occupied-factor and multi-plan contraction descriptions."""
    terms = parse_metadata(program)
    if kind in ("ace", "linear"):
        return tuple((0, mapping, role, slot, 0) for mapping, role, slot in terms)
    if kind == "bilinear":
        return tuple(
            (plan, mapping, role, slot, 0) for plan, mapping, role, slot in terms
        )
    return tuple((0, *term) for term in terms)


def encode_program(kind, terms):
    """Retain the native operator's existing program format."""
    if kind in ("ace", "linear"):
        return repr(tuple((mapping, role, slot) for _, mapping, role, slot, _ in terms))
    if kind == "bilinear":
        return repr(tuple(term[:4] for term in terms))
    return repr(tuple(term[1:] for term in terms))


def polynomial_adjoint(program, operands, cotangents, needed, kind="channel"):
    """Transpose a polynomial program while retaining its occupied factor roles."""
    values = list(operands)
    seeds = {}
    for slot, gradient in enumerate(cotangents):
        if gradient is not None:
            seeds[slot] = len(values)
            values.append(gradient)
    terms, destinations = [], {}
    for plan, mapping, output, slot, required in program_terms(kind, program):
        if slot not in seeds:
            continue
        replacement = list(mapping)
        replacement[output] = seeds[slot]
        for role, index in enumerate(mapping):
            if role != output and needed[index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append(
                    (
                        plan,
                        tuple(replacement),
                        role,
                        destination,
                        required | (1 << output),
                    )
                )
    return encode_program(kind, terms), values, destinations


def polynomial_tangent(program, num_operands, kind="channel"):
    """Differentiate occupied factor roles with one tangent per operand."""
    terms = []
    for plan, mapping, output, slot, required in program_terms(kind, program):
        for role, index in enumerate(mapping):
            if role != output:
                replacement = list(mapping)
                replacement[role] = num_operands + index
                terms.append(
                    (plan, tuple(replacement), output, slot, required | (1 << role))
                )
    return encode_program(kind, terms)


def evaluate_polynomial(kind, metadata, program, node_type, operands):
    """Use explicit Torch products inside functional transformations."""
    if kind == "channel":
        from .kernels.torch.channel_product import channel_product

        return channel_product(metadata, program, operands)
    if kind == "symmetric":
        from .kernels.torch.symmetric import evaluate_torch

        return evaluate_torch(metadata, program, node_type, operands)
    if kind in ("ace", "bilinear"):
        from .kernels.torch.ace import contract

        return contract(kind, metadata, program, node_type, operands)
    if kind == "linear":
        from .kernels.torch.linear import contract

        return evaluate_linear(contract, metadata, program, node_type, operands)
    raise ValueError(f"Unknown polynomial kind {kind!r}.")


def evaluate_linear(contract, metadata, program, node_type, operands):
    """Sum transposed indexed-linear terms into their declared output slots."""
    terms = parse_metadata(program)
    outputs = [None] * (1 + max(slot for _, _, slot in terms))
    for mapping, role, slot in terms:
        value = contract(metadata, role, node_type, *(operands[i] for i in mapping))
        outputs[slot] = value if outputs[slot] is None else outputs[slot] + value
    return outputs


@torch.compiler.allow_in_graph
class NativePolynomial(torch.autograd.Function):
    """Native primal/adjoints with analytic Torch JVPs and batching rules."""

    @staticmethod
    def forward(kind, metadata, program, node_type, *operands):
        if kind == "channel":
            return tuple(
                torch.ops.eqx.local_channel_product(metadata, program, list(operands))
            )
        if kind == "linear":
            return tuple(
                evaluate_linear(
                    torch.ops.eqx.element_linear, metadata, program, node_type, operands
                )
            )
        name = {
            "symmetric": "symmetric_contract",
            "ace": "ace_contract",
            "bilinear": "bilinear_ace",
        }[kind]
        return tuple(
            getattr(torch.ops.eqx, name)(metadata, program, node_type, list(operands))
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.kind, ctx.kernel_metadata, ctx.program, node_type, *operands = inputs
        ctx.save_for_backward(node_type, *operands)
        ctx.save_for_forward(node_type, *operands)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *cotangents):
        node_type, *operands = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[4:])
        program, values, destinations = polynomial_adjoint(
            ctx.program, operands, cotangents, needed, ctx.kind
        )
        gradients = [None] * len(operands)
        if destinations:
            outputs = native_polynomial(
                ctx.kind, ctx.kernel_metadata, program, node_type, *values
            )
            for index, slot in destinations.items():
                gradients[index] = outputs[slot]
        return None, None, None, None, *gradients

    @staticmethod
    def jvp(
        ctx, kind_tangent, metadata_tangent, program_tangent, type_tangent, *tangents
    ):
        node_type, *operands = ctx.saved_tensors
        tangents = tuple(
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(operands, tangents)
        )
        program = polynomial_tangent(ctx.program, len(operands), ctx.kind)
        return tuple(
            evaluate_polynomial(
                ctx.kind,
                ctx.kernel_metadata,
                program,
                node_type,
                (*operands, *tangents),
            )
        )

    @staticmethod
    def vmap(info, in_dims, kind, metadata, program, node_type, *operands):
        def function(types, *values):
            return tuple(evaluate_polynomial(kind, metadata, program, types, values))

        outputs = torch.vmap(function, in_dims=in_dims[3:], randomness=info.randomness)(
            node_type, *operands
        )
        return outputs, (0,) * len(outputs)


@torch.compiler.allow_in_graph
def native_polynomial(kind, metadata, program, node_type, *operands):
    """Retain native derivatives and expose nested forward AD to ordinary ops."""
    validate_uniform(operands)
    if nested_forward_ad():
        return tuple(evaluate_polynomial(kind, metadata, program, node_type, operands))
    return NativePolynomial.apply(kind, metadata, program, node_type, *operands)
