"""AOT-visible native scalar geometry and recursively generated transpose kernels."""

from functools import lru_cache

import torch

from ....utils.autograd import forward_ad_active
from ....utils.metadata import parse_metadata
from ..scalar import rotation_program
from ..torch import scalar as torch_kernel
from .pointwise import launch


@lru_cache(maxsize=128)
def generated_kernel(nodes, outputs, widths, dtype):
    """Generate scalar CUDA expressions for an alignment chart or its adjoint."""
    scalar = "double" if dtype == torch.float64 else "float"
    prefix = "d" if dtype == torch.float64 else "f"
    parameters = [f"const T* p{i}" for i in range(len(widths))]
    parameters.extend(f"T* out{i}" for i in range(len(outputs)))
    lines = []
    for index, (op, *args) in enumerate(nodes):
        if op == "constant":
            expression = f"T({args[0]:.17g})"
        elif op == "input":
            expression = f"p{args[0]}[row*{widths[args[0]]}+{args[1]}]"
        else:
            names = [f"v{k}" for k in args]
            if op == "select":
                expression = f"({names[0]} ? {names[1]} : {names[2]})"
            elif op == "neg":
                expression = f"-{names[0]}"
            elif op in ("sqrt", "exp"):
                expression = f"{op}({names[0]})"
            elif op in ("add", "mul"):
                expression = f"__{prefix}{op}_rn({names[0]},{names[1]})"
            else:
                expression = (
                    f"{names[0]} "
                    + {"div": "/", "lt": "<", "le": "<="}[op]
                    + f" {names[1]}"
                )
        lines.append(f"auto v{index}={expression};")
    for group, output in enumerate(outputs):
        for column, node in enumerate(output):
            lines.append(f"out{group}[row*{len(output)}+{column}]=v{node};")
    return f"""
            using T={scalar}; using I=long long;
        extern "C" __global__ void run({",".join(parameters)},I count) {{
            for(I row=I(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=I(gridDim.x)*blockDim.x) {{
                {"".join(lines)}
            }}
        }}
    """


@torch.library.custom_op("eqx::scalar_geometry", mutates_args=(), device_types="cuda")
def contract(key: str, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    axis, eps, quaternion, derivative = parse_metadata(key)
    nodes, outputs, widths = rotation_program(
        axis,
        eps,
        quaternion,
        "float" if inputs[0].dtype == torch.float32 else "double",
        derivative,
    )
    inputs = [value.contiguous() for value in inputs]
    count = inputs[0].shape[0]
    results = [inputs[0].new_empty((count, len(group))) for group in outputs]
    code = generated_kernel(nodes, outputs, widths, inputs[0].dtype)
    launch(code, (*inputs, *results), (count,), count, threads=128)
    return results


@contract.register_fake
def fake(key, inputs):
    axis, eps, quaternion, derivative = parse_metadata(key)
    _, outputs, _ = rotation_program(
        axis,
        eps,
        quaternion,
        "float" if inputs[0].dtype == torch.float32 else "double",
        derivative,
    )
    return [inputs[0].new_empty((inputs[0].shape[0], len(group))) for group in outputs]


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
