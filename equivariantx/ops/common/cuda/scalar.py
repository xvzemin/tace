"""AOT-visible native scalar geometry and recursively generated transpose kernels."""

import hashlib
import linecache
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra.cuda import libdevice

from ....utils.autograd import forward_ad_active
from ....utils.metadata import parse_metadata
from ..scalar import rotation_program
from ..torch import scalar as torch_kernel


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def generated_kernel(nodes, outputs, widths):
    """Emit one scalar expression per statement instead of growing Triton tuples."""
    inputs = [f"p{i}" for i in range(len(widths))]
    results = [f"out{i}" for i in range(len(outputs))]
    lines = [
        "def geometry("
        + ", ".join((*inputs, *results, "count", "BLOCK: tl.constexpr"))
        + "):",
        "    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = row < count",
        "    dtype: tl.constexpr = p0.dtype.element_ty",
    ]
    for index, (op, *args) in enumerate(nodes):
        if op == "constant":
            value = f"tl.full((BLOCK,), {args[0]!r}, dtype)"
        elif op == "input":
            value = (
                f"tl.load(p{args[0]} + row * {widths[args[0]]} + {args[1]}, mask, 0)"
            )
        else:
            names = [f"v{k}" for k in args]
            if op == "select":
                value = "tl.where(" + ", ".join(names) + ")"
            elif op == "neg":
                value = "-" + names[0]
            elif op in ("sqrt", "exp"):
                value = f"libdevice.{op}({names[0]})"
            else:
                operator = {"add": "+", "mul": "*", "div": "/", "lt": "<", "le": "<="}[
                    op
                ]
                value = f"{names[0]} {operator} {names[1]}"
        lines.append(f"    v{index} = {value}")
    for group, output in enumerate(outputs):
        for column, node in enumerate(output):
            lines.append(
                f"    tl.store(out{group} + row * {len(output)} + {column}, v{node}, mask)"
            )
    source = "\n".join(lines) + "\n"
    filename = "<eqx_scalar_" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"tl": tl, "libdevice": libdevice, "__name__": __name__}
    exec(compile(source, filename, "exec"), namespace)
    return triton.jit(namespace["geometry"])


@triton_op("eqx::scalar_geometry", mutates_args=())
def contract(key: str, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    axis, eps, quaternion, derivative = parse_metadata(key)
    nodes, outputs, widths = rotation_program(
        axis,
        eps,
        quaternion,
        "float" if inputs[0].dtype == torch.float32 else "double",
        derivative,
    )
    if len(nodes) > 2048:
        return list(torch_kernel.primal(axis, eps, quaternion, derivative, *inputs))
    inputs = [value.contiguous() for value in inputs]
    count = inputs[0].shape[0]
    results = [inputs[0].new_empty((count, len(group))) for group in outputs]
    if count:
        wrap_triton(generated_kernel(nodes, outputs, widths))[
            (triton.cdiv(count, 128),)
        ](
            *inputs,
            *results,
            count,
            128,
            enable_fp_fusion=False,
        )
    return results


def setup_context(ctx, inputs, output):
    key, values = inputs
    ctx.key = parse_metadata(key)
    ctx.save_for_backward(*values)
    ctx.output_widths = tuple(value.shape[-1] for value in output)
    ctx.set_materialize_grads(False)


def backward(ctx, gradients):
    axis, eps, quaternion, parent = ctx.key
    inputs = ctx.saved_tensors
    active = tuple(i for i, needed in enumerate(ctx.needs_input_grad[1]) if needed)
    result = [None] * len(inputs)
    if active:
        gradients = [
            value
            if value is not None
            else inputs[0].new_zeros((inputs[0].shape[0], width))
            for value, width in zip(gradients, ctx.output_widths)
        ]
        values = contract(
            repr((axis, eps, quaternion, (parent, active))), [*inputs, *gradients]
        )
        for index, value in zip(active, values):
            result[index] = value
    return None, result


contract.register_autograd(backward, setup_context=setup_context)


def program_key(axis, eps, quaternion, derivative):
    """Specialize immutable chart options before dynamic shape tracing."""
    if eps == 1e-7:
        return (
            "("
            + str(axis)
            + ", 1e-07, "
            + str(quaternion)
            + ", "
            + repr(derivative)
            + ")"
        )
    return repr((axis, eps, quaternion, derivative))


def evaluate(axis, eps, quaternion, derivative, *inputs):
    if forward_ad_active():
        return torch_kernel.primal(axis, eps, quaternion, derivative, *inputs)
    return tuple(contract(program_key(axis, eps, quaternion, derivative), list(inputs)))
