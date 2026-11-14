"""Indexed bilinear contractions and their transposes."""

import torch

from ...utils.outputs import linear_output
from .polynomial import native_polynomial


@torch.library.custom_op("eqx::element_linear", mutates_args=(), device_types="cuda")
def contract_cuda(
    metadata: str,
    role: int,
    node_type: torch.Tensor,
    features: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Contract two operands into the selected feature or weight role."""
    from .. import get_kernel

    return get_kernel("element_linear", "cuda")(
        metadata, role, node_type, features, weight, output
    )


contract_cuda.register_fake(linear_output)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, ctx.role, node_type, *operands = inputs
    ctx.save_for_backward(node_type, *operands)


def backward(ctx, grad_output):
    node_type, *operands = ctx.saved_tensors
    operands[ctx.role] = grad_output
    gradients = [None, None, None]
    for role in range(3):
        if role != ctx.role and ctx.needs_input_grad[role + 3]:
            gradients[role] = contract(ctx.kernel_metadata, role, node_type, *operands)
    return None, None, None, *gradients


contract_cuda.register_autograd(backward, setup_context=setup_context)


def contract(metadata, role, node_type, features, weight, output):
    """Apply native indexed maps with explicit functional-transform rules."""
    node_type = torch.arange(weight.shape[0], device=node_type.device)[node_type]
    program = repr((((0, 1, 2), role, 0),))
    return native_polynomial(
        "linear", metadata, program, node_type, features, weight, output
    )[0]


def linear_program(program, inputs, weight, bias, description):
    """Encode grouped channel GEMMs without materializing edge intermediates."""
    input_shapes, output_shapes, paths, biases, groups = description
    uniform = bool(paths) and all(path[3] == paths[0][3] for path in paths)
    if uniform:
        weight = program.scale(weight, paths[0][3])
    matrices = [
        program.scale(
            program.slice(weight, start, input_shapes[i][1] * output_shapes[j][1]),
            1 if uniform else scale,
        )
        for i, j, start, scale in paths
    ]

    def columns(values, rows, widths):
        if len(values) == 1:
            return values[0]
        return program.transpose(
            program.concatenate(
                [
                    program.transpose(value, rows, width)
                    for value, width in zip(values, widths)
                ]
            ),
            sum(widths),
            rows,
        )

    outputs = [None] * len(output_shapes)
    for entries in groups:
        selected = entries[0][1]
        if not selected:
            for j, _ in entries:
                dim, mul = output_shapes[j]
                outputs[j] = program.constant(dim * mul, 0)
            continue
        sources = [paths[k][0] for k in selected]
        widths = [input_shapes[i][1] for i in sources]
        dim = output_shapes[entries[0][0]][0]
        out_widths = [output_shapes[j][1] for j, _ in entries]
        value = columns([inputs[i] for i in sources], dim, widths)
        matrix = program.concatenate(
            [
                columns(
                    [matrices[indices[row]] for _, indices in entries], ci, out_widths
                )
                for row, ci in enumerate(widths)
            ]
        )
        result = program.matmul(value, matrix, dim, sum(widths), sum(out_widths))
        offset = 0
        for (j, _), width in zip(entries, out_widths):
            outputs[j] = program.concatenate(
                [
                    program.slice(result, row * sum(out_widths) + offset, width)
                    for row in range(dim)
                ]
            )
            offset += width
    for j, start, size in biases:
        value = program.slice(bias, start, size)
        value = program.gather(value, tuple(range(size)) * output_shapes[j][0])
        outputs[j] = program.binary("add", outputs[j], value)
    return outputs
