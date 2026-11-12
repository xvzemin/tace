"""Batched Torch polynomials with recursively transposed derivatives."""

from collections import defaultdict
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import nested_forward_ad, torch_gradient_mask
from ....utils.outputs import symmetric_outputs
from ... import config
from ...polynomial import polynomial_adjoint, polynomial_groups, polynomial_tangent


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def polynomial_batches(metadata, program):
    """Batch terms sharing operand slots and the element-indexing convention."""
    batches = defaultdict(list)
    for (slot, reduce, component), expressions in polynomial_groups(metadata, program):
        for factors, coefficient in expressions:
            signature = tuple((operand, indexed) for operand, _, indexed in factors)
            batches[slot, reduce, signature].append(
                (component, tuple(index for _, index, _ in factors), coefficient)
            )
    return tuple((key, tuple(terms)) for key, terms in batches.items())


@lru_cache(maxsize=128)
def polynomial_indices(metadata, program, device, dtype):
    """Place model-constant sparse indices and coefficients on the operand device."""
    return tuple(
        (
            torch.tensor([term[0] for term in terms], device=device, dtype=torch.long),
            tuple(
                torch.tensor(
                    [term[1][i] for term in terms], device=device, dtype=torch.long
                )
                for i in range(len(signature))
            ),
            torch.tensor([term[2] for term in terms], device=device, dtype=dtype),
        )
        for (_, _, signature), terms in polynomial_batches(metadata, program)
    )


def evaluate_torch(metadata, program, node_type, operands):
    """Evaluate batched sparse products with ordinary differentiable Torch ops."""
    outputs = [
        torch.zeros_like(value)
        for value in symmetric_outputs(metadata, program, node_type, operands)
    ]
    batches = polynomial_batches(metadata, program)
    tracing = (
        torch.compiler.is_compiling()
        or is_fake(operands[0])
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    )
    index_function = polynomial_indices.__wrapped__ if tracing else polynomial_indices
    indices = index_function(metadata, program, operands[0].device, operands[0].dtype)
    selected = {}
    for ((slot, reduce, signature), _), (destination, columns, coefficient) in zip(
        batches, indices
    ):
        value = coefficient[None, :, None]
        for (operand, indexed), column in zip(signature, columns):
            if (operand, indexed) not in selected:
                selected[operand, indexed] = (
                    operands[operand][node_type.long()]
                    if indexed
                    else operands[operand]
                )
            value = value * selected[operand, indexed].index_select(1, column)
        value = value.expand(
            node_type.shape[0], destination.shape[0], operands[0].shape[-1]
        )
        if reduce:
            output = outputs[slot]
            index = node_type[:, None] * output.shape[1] + destination[None, :]
            outputs[slot] = (
                output.flatten(0, 1)
                .index_add(0, index.flatten().long(), value.flatten(0, 1))
                .reshape(output.shape)
            )
        else:
            outputs[slot] = outputs[slot].index_add(1, destination, value)
    used = {operand for (_, _, signature), _ in batches for operand, _ in signature}
    unused = [value[..., :0].sum() for i, value in enumerate(operands) if i not in used]
    if unused:
        zero = sum(unused)
        outputs = [value + zero for value in outputs]
    return outputs


@torch.compiler.allow_in_graph
class SymmetricProgram(torch.autograd.Function):
    """Recompute polynomial terms while combining equivalent derivative factors."""

    generate_vmap_rule = True

    @staticmethod
    def forward(metadata, program, node_type, *operands):
        return tuple(evaluate_torch(metadata, program, node_type, operands))

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.kernel_metadata, ctx.program, node_type, *operands = inputs
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(node_type, *operands)
        ctx.save_for_forward(node_type, *operands)

    @staticmethod
    def backward(ctx, *cotangents):
        node_type, *operands = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[3:])
        program, values, destinations = polynomial_adjoint(
            ctx.program, operands, cotangents, needed
        )
        gradients = [None] * len(operands)
        if destinations:
            results = evaluate_recomputed(
                ctx.kernel_metadata, program, node_type, *values
            )
            for index, slot in destinations.items():
                gradients[index] = results[slot]
        gradients = [
            torch.zeros_like(value) if required and grad is None else grad
            for value, required, grad in zip(operands, needed, gradients)
        ]
        return None, None, None, *gradients

    @staticmethod
    def jvp(ctx, metadata_tangent, program_tangent, type_tangent, *tangents):
        node_type, *operands = ctx.saved_tensors
        tangents = tuple(
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(operands, tangents)
        )

        program = polynomial_tangent(ctx.program, len(operands))
        return tuple(
            evaluate_torch(
                ctx.kernel_metadata, program, node_type, (*operands, *tangents)
            )
        )


def contract(metadata, program, node_type, operands):
    """Select stored activations or recursively recomputed polynomial adjoints."""
    if config.TORCH_RECOMPUTE:
        return list(evaluate_recomputed(metadata, program, node_type, *operands))
    return evaluate_torch(metadata, program, node_type, operands)


@torch.compiler.allow_in_graph
def evaluate_recomputed(metadata, program, node_type, *operands):
    """Keep repeated forward transforms in the ordinary polynomial graph."""
    if nested_forward_ad():
        return tuple(evaluate_torch(metadata, program, node_type, operands))
    return SymmetricProgram.apply(metadata, program, node_type, *operands)
