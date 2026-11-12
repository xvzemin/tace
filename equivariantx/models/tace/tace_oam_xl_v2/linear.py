"""Whole-irrep linear programs with shared exact mixed-derivative schedules."""

from collections import defaultdict

import torch

from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.metadata import parse_metadata


def linear_outputs(metadata, program, operands):
    """Allocate fresh contiguous results for a static bilinear program."""
    terms = parse_metadata(program)
    results = [None] * (1 + max(slot for _, _, _, slot in terms))
    for _, mapping, role, slot in terms:
        if results[slot] is None:
            results[slot] = torch.empty_like(
                operands[mapping[role]], memory_format=torch.contiguous_format
            )
    return results


def linear_groups(metadata, program):
    """Collect all contributions to an output irrep before matrix evaluation."""
    paths = parse_metadata(metadata)[-1]
    groups = [defaultdict(list) for _ in range(3)]
    for path, mapping, role, slot in parse_metadata(program):
        groups[role][path, slot].append(mapping)
    return tuple(
        tuple(
            (paths[path], slot, tuple(terms)) for (path, slot), terms in group.items()
        )
        for group in groups
    )


def reference_linear(metadata, program, operands):
    """FP64 implementation of the same complete multilinear program."""
    layout_x, layout_y, width_x, width_y, paths = parse_metadata(metadata)
    results = linear_outputs(metadata, program, operands)
    for result in results:
        result.zero_()
    for path, (x, w, y), role, slot in parse_metadata(program):
        ix, ci, iy, co, dim, scale = paths[path]
        rows = operands[x].size(0)

        def view(value, start, channels, layout):
            block = value[:, start : start + channels * dim]
            return (
                block.reshape(rows, dim, channels)
                if layout == "ir_mul"
                else block.reshape(rows, channels, dim).transpose(1, 2)
            )

        if role == 2:
            value = view(operands[x], ix, ci, layout_x) @ operands[w]
            view(results[slot], iy, co, layout_y).add_(value * scale)
        elif role == 0:
            value = (view(operands[y], iy, co, layout_y) * scale) @ operands[w].T
            view(results[slot], ix, ci, layout_x).add_(value)
        else:
            a = view(operands[x], ix, ci, layout_x).reshape(-1, ci)
            b = view(operands[y], iy, co, layout_y).reshape(-1, co)
            results[slot].add_(a.T @ (b * scale))
    return results


def run_linear(metadata, program, operands):
    """Execute grouped native GEMMs without full-node layout temporaries."""
    if (
        operands[0].dtype != torch.float32
        or not operands[0].is_cuda
        or torch.cuda.get_device_capability(operands[0].device)[0] < 9
    ):
        return reference_linear(metadata, program, operands)
    from .linear_kernel import node_program, weight_program, weight_reduce

    layout_x, layout_y, width_x, width_y, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    groups = linear_groups(metadata, program)
    results = linear_outputs(metadata, program, operands)
    # A mixed derivative seeded at only some weight matrices can leave entire
    # irreps untouched. Those columns must be explicit zeros.
    for role in (0, 2):
        coverage = defaultdict(int)
        for path, slot, _ in groups[role]:
            coverage[slot] += path[1 if role == 0 else 3] * path[4]
        for slot, count in coverage.items():
            if count != results[slot].size(1):
                results[slot].zero_()
    if not operands[terms[0][1][0]].size(0):
        for result in results:
            result.zero_()
        return results
    rows = operands[terms[0][1][0]].size(0)
    read = {
        index
        for _, mapping, output, _ in terms
        for role, index in enumerate(mapping)
        if role != output
    }
    operands = tuple(x.contiguous() if i in read else x for i, x in enumerate(operands))
    precision = "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"
    for role in (0, 2):
        group = groups[role]
        if not group:
            continue
        grid = sum(
            ((rows * path[4] + 63) // 64) * ((path[1 if role == 0 else 3] + 63) // 64)
            for path, _, _ in group
        )
        node_program[(grid,)](
            operands,
            tuple(results),
            rows,
            group,
            layout_x,
            layout_y,
            width_x,
            width_y,
            role,
            precision,
        )
    group = groups[1]
    if group:
        partial = tuple(
            operands[0].new_empty((8, path[1], path[3])) for path, _, _ in group
        )
        grid = sum(
            ((path[1] + 63) // 64) * ((path[3] + 63) // 64) for path, _, _ in group
        )
        weight_program[(grid, 8)](
            operands,
            partial,
            rows,
            group,
            layout_x,
            layout_y,
            width_x,
            width_y,
            precision,
        )
        grid = sum((path[1] * path[3] + 255) // 256 for path, _, _ in group)
        weight_reduce[(grid,)](partial, tuple(results), group)
    return results


@torch.library.custom_op("eqx_oam_xl::linear", mutates_args=())
def linear_contract(
    metadata: str, program: str, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Apply disjoint irrep matrices and their explicit bilinear adjoints."""
    return run_linear(metadata, program, operands)


linear_contract.register_fake(linear_outputs)


def linear_context(ctx, inputs, output):
    metadata, program, operands = inputs
    ctx.kernel_metadata, ctx.program = metadata, parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*operands)


def linear_backward(ctx, grad_outputs):
    operands = list(ctx.saved_tensors)
    required = gradient_mask(operands, ctx.needs_input_grad[2], offset=0)
    values = list(operands)
    cotangents = {}
    for slot, value in enumerate(grad_outputs):
        if value is not None:
            cotangents[slot] = len(values)
            values.append(value)
    terms, destinations = [], {}
    for path, mapping, output, slot in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role == output or not required[index]:
                continue
            destination = destinations.setdefault(index, len(destinations))
            terms.append((path, tuple(replacement), role, destination))
    gradients = [None] * len(operands)
    if terms:
        outputs = linear_contract(ctx.kernel_metadata, repr(tuple(terms)), values)
        for index, slot in destinations.items():
            gradients[index] = outputs[slot]
    return None, None, gradients


linear_contract.register_autograd(linear_backward, setup_context=linear_context)


def irrep_linear(module, inputs):
    """Evaluate a validated one-to-one irrep map with live ParameterList weights."""
    weights = list(module.weight)
    slices_in, slices_out = module.irreps_in.slices(), module.irreps_out.slices()
    paths = []
    for ins in module.linear.instructions:
        ci, ir = module.irreps_in[ins.i_in]
        co, _ = module.irreps_out[ins.i_out]
        paths.append(
            (
                slices_in[ins.i_in].start,
                ci,
                slices_out[ins.i_out].start,
                co,
                ir.dim,
                ins.path_weight,
            )
        )
    metadata = repr(
        (
            getattr(module, "input_layout", "mul_ir"),
            "mul_ir",
            module.irreps_in.dim,
            module.irreps_out.dim,
            tuple(paths),
        )
    )
    template = inputs.new_empty(1).expand(inputs.size(0), module.irreps_out.dim)
    operands = [inputs, *weights, template]
    program = tuple((i, (0, i + 1, len(operands) - 1), 2, 0) for i in range(len(paths)))
    result = linear_contract(metadata, repr(program), operands)[0]
    if module.bias is not None:
        pieces, offset = [], 0
        for (co, ir), section in zip(module.irreps_out, slices_out):
            value = result[:, section]
            if ir.is_scalar():
                value = value + module.bias[offset : offset + co]
                offset += co
            pieces.append(value)
        result = torch.cat(pieces, -1)
    return result
