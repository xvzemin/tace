"""Scalar geometry with an operand-only tape and recursive exact transposes."""

import operator

import torch

from ....utils.autograd import forward_ad_active, torch_gradient_mask
from ..scalar import rotation_program


def primal(axis, eps, quaternion, derivative, *inputs):
    """Evaluate pruned expressions using ordinary differentiable Torch ops."""
    nodes, outputs, _ = rotation_program(
        axis,
        eps,
        quaternion,
        "float" if inputs[0].dtype == torch.float32 else "double",
        derivative,
    )
    values = []
    for op, *args in nodes:
        if op == "input":
            value = inputs[args[0]][:, args[1]]
        elif op == "constant":
            value = args[0]
        else:
            arguments = [values[k] for k in args]
            if op == "select":
                value = (
                    torch.where(*arguments)
                    if isinstance(arguments[0], torch.Tensor)
                    else arguments[1]
                    if arguments[0]
                    else arguments[2]
                )
            elif op in ("sqrt", "exp"):
                value = getattr(torch, op)(*arguments)
            else:
                value = getattr(operator, "truediv" if op == "div" else op)(*arguments)
        values.append(value)
    zero = sum(value[..., :0].sum() for value in inputs)
    return tuple(
        torch.stack(
            [
                values[k]
                if isinstance(values[k], torch.Tensor)
                else inputs[0].new_full((inputs[0].shape[0],), values[k])
                for k in group
            ],
            dim=-1,
        )
        + zero
        for group in outputs
    )


@torch.compiler.allow_in_graph
class ScalarAdjoint(torch.autograd.Function):
    """Retain operands and recompute each requested scalar transpose."""

    generate_vmap_rule = True

    @staticmethod
    def forward(axis, eps, quaternion, derivative, *inputs):
        return primal(axis, eps, quaternion, derivative, *inputs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.axis, ctx.eps, ctx.quaternion, ctx.derivative, *values = inputs
        ctx.save_for_backward(*values)
        ctx.output_widths = tuple(value.shape[-1] for value in output)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *cotangents):
        values = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[4:])
        active = tuple(i for i, need in enumerate(needed) if need)
        result = [None] * len(values)
        if active:
            gradients = tuple(
                value
                if value is not None
                else values[0].new_zeros((values[0].shape[0], width))
                for value, width in zip(cotangents, ctx.output_widths)
            )
            derivatives = evaluate(
                ctx.axis,
                ctx.eps,
                ctx.quaternion,
                (ctx.derivative, active),
                *values,
                *gradients,
            )
            for index, value in zip(active, derivatives):
                result[index] = value
        return None, None, None, None, *result


@torch.compiler.allow_in_graph
def evaluate(axis, eps, quaternion, derivative, *inputs):
    """Use explicit primitives for functional transforms and an exact adjoint otherwise."""
    if forward_ad_active():
        return primal(axis, eps, quaternion, derivative, *inputs)
    return ScalarAdjoint.apply(axis, eps, quaternion, derivative, *inputs)
