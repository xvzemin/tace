"""Symmetric polynomial schedules shared by execution providers."""

from collections import Counter, defaultdict
from functools import lru_cache

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ...utils.autograd import nested_forward_ad, torch_gradient_mask
from ...utils.metadata import (
    compact_metadata,
    independent_channel_groups,
    parse_metadata,
)
from ...utils.scalar import ScalarProgram
from .. import config
from ..precision import validate_uniform
from .compilation import polynomial_work, should_inline, wide_ace, wide_symmetric

EXTERNAL_POLYNOMIALS = {}


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


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def polynomial_prefix(groups):
    """Factor sparse expressions and share identical scalar products."""
    graph = ScalarProgram()

    def product(factors):
        value = graph.constant(1)
        for factor in factors:
            value = graph.add("mul", value, graph.add("input", *factor))
        return value

    def expression(terms):
        counts = Counter(factor for factors, _ in terms for factor in set(factors))
        if not counts or max(counts.values()) == 1:
            value = graph.constant(0)
            for factors, coefficient in terms:
                value = graph.add(
                    "add",
                    value,
                    graph.add("mul", graph.constant(coefficient), product(factors)),
                )
            return value
        factor = max(counts, key=lambda item: (counts[item], item[2], item))
        selected, remainder = [], []
        for factors, coefficient in terms:
            if factor in factors:
                rest = list(factors)
                rest.remove(factor)
                selected.append((tuple(rest), coefficient))
            else:
                remainder.append((factors, coefficient))
        value = graph.add("mul", product((factor,)), expression(selected))
        return graph.add("add", value, expression(remainder)) if remainder else value

    roots = tuple((key, expression(terms)) for key, terms in groups)
    return tuple(graph.nodes), roots


@lru_cache(maxsize=256)
def program_terms(kind, program):
    """Normalize occupied-factor and multi-plan contraction descriptions."""
    terms = parse_metadata(program)
    if kind in ("ace", "linear"):
        return tuple((0, mapping, role, slot, 0) for mapping, role, slot in terms)
    if kind == "indexed_plan":
        return tuple(
            (plan, mapping, role, slot, 0) for plan, mapping, role, slot in terms
        )
    return tuple((0, *term) for term in terms)


def encode_program(kind, terms):
    """Retain the native operator's existing program format."""
    if kind in ("ace", "linear"):
        return repr(tuple((mapping, role, slot) for _, mapping, role, slot, _ in terms))
    if kind == "indexed_plan":
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
        from .torch.channel_product import channel_product

        return channel_product(metadata, program, operands)
    if kind in ("symmetric", "symmetric_matrix"):
        from .torch.symmetric import evaluate_torch

        return evaluate_torch(metadata, program, node_type, operands)
    if kind in ("ace", "indexed_plan"):
        from .torch.ace import contract

        return contract(kind, metadata, program, node_type, operands)
    if kind == "linear":
        from .torch.linear import contract

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


def eager_kernel(kind, node_type, operands):
    """Select a native eager contraction outside Torch dispatch transforms."""
    return (
        kind in ("ace", "channel")
        and operands[0].is_cuda
        and not torch.compiler.is_compiling()
        and not torch.jit.is_tracing()
        and not torch._C._are_functorch_transforms_active()
        and not torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Python
        )
        and not torch._C._is_torch_function_mode_enabled()
        and (node_type is None or type(node_type) is torch.Tensor)
        and all(type(x) in (torch.Tensor, torch.nn.Parameter) for x in operands)
    )


def run_eager_kernel(kind, metadata, program, node_type, operands):
    from .. import get_kernel

    if kind == "channel":
        return tuple(
            get_kernel("local_channel_product", "cuda")(metadata, program, operands)
        )
    return tuple(
        get_kernel("ace_contract", "cuda")(metadata, program, node_type, operands)
    )


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def joint_channel_program(metadata, program, dtype):
    """The CUDA program has one writer for every requested output coordinate."""
    if independent_channel_groups(metadata) is None:
        return False
    # This is cached static code generation, not a kernel build or launch.
    from .cuda.channel_product import joint_product_source

    return joint_product_source(metadata, program, dtype) is not None


def native_joint_channel(metadata, program, operands):
    x = operands[0]
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float32, torch.float64)
        and joint_channel_program(metadata, program, x.dtype)
        and all(
            statically_known_true(t.shape[0] == 0)
            or statically_known_true(t.shape[0] == x.shape[0])
            for t in operands
        )
    )


@torch.compiler.allow_in_graph
class NativePolynomial(torch.autograd.Function):
    """Native primal/adjoints with analytic Torch JVPs and batching rules."""

    @staticmethod
    def forward(kind, metadata, program, node_type, *operands):
        if eager_kernel(kind, node_type, operands):
            return run_eager_kernel(kind, metadata, program, node_type, operands)
        if not operands[0].is_cuda:
            return tuple(
                value.view_as(value)
                for value in evaluate_polynomial(
                    kind, metadata, program, node_type, operands
                )
            )
        if kind == "symmetric_matrix" or (
            kind == "symmetric" and config.SYMMETRIC_NATIVE
        ):
            return tuple(
                value.view_as(value)
                for value in torch.ops.eqx.symmetric_contract(
                    metadata, program, node_type, list(operands)
                )
            )
        if kind == "linear" and config.LINEAR_NATIVE and operands[0].is_cuda:
            return tuple(
                evaluate_linear(
                    torch.ops.eqx.element_linear, metadata, program, node_type, operands
                )
            )
        if should_inline(
            config.POLYNOMIAL_INLINE, polynomial_work, kind, metadata, program
        ) and not (
            (kind == "symmetric" and wide_symmetric(metadata, operands[0].shape[-1]))
            or (kind == "ace" and wide_ace(metadata, node_type.shape[0]))
            or (kind == "channel" and native_joint_channel(metadata, program, operands))
        ):
            return tuple(
                value.view_as(value)
                for value in evaluate_polynomial(
                    kind, metadata, program, node_type, operands
                )
            )
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
        if kind in EXTERNAL_POLYNOMIALS:
            op, _ = EXTERNAL_POLYNOMIALS[kind]
            return tuple(op(metadata, program, node_type, list(operands)))
        name = {
            "symmetric": "symmetric_contract",
            "ace": "ace_contract",
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
    metadata = metadata if kind == "indexed_plan" else compact_metadata(metadata)
    program = compact_metadata(program)
    if (
        (not torch.is_grad_enabled() or not any(x.requires_grad for x in operands))
        and torch.autograd.forward_ad._current_level < 0
        and eager_kernel(kind, node_type, operands)
    ):
        return run_eager_kernel(kind, metadata, program, node_type, operands)
    if nested_forward_ad():
        return tuple(evaluate_polynomial(kind, metadata, program, node_type, operands))
    return NativePolynomial.apply(kind, metadata, program, node_type, *operands)
