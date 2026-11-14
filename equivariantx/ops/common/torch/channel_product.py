"""Sparse channel polynomials using differentiable gathers and reductions."""

import math
from collections import defaultdict
from functools import lru_cache

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ....utils.metadata import channel_product_layout, parse_metadata
from ... import config


def cpu_slices(value):
    """Keep the measured slice schedule within CPU joint compilation."""
    return (
        value.device.type == "cpu"
        and config.TORCH_CHANNEL_SLICES
        and config.POLYNOMIAL_INLINE
        and not config.TORCH_REFERENCE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


@lru_cache(maxsize=128)
def minimum_width(metadata):
    """Return the narrowest static channel run, or zero for empty programs."""
    _, rows = channel_product_layout(metadata)
    return min((width for _, _, width, _ in rows), default=0)


def use_slices(module, x, y, weight):
    """Select measured CPU schedules while preserving legacy checkpoints."""
    metadata = getattr(module, "kernel_metadata", None)
    if (
        getattr(module, "backend", "torch") != "torch"
        or not metadata
        or not cpu_slices(x)
    ):
        return False
    width = minimum_width(metadata)
    if width < config.CHANNEL_SLICES_MIN_WIDTH:
        return False
    if (
        module.contraction.layout == "ir_mul"
        and module.shared_weights
        and width == 1024
        and all(ins.connection_mode == "u1u" for ins in module.instructions)
    ):
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        # The ordinary tree is faster for this static small-batch shape.
        # Symbolic counts retain one program without a runtime shape guard.
        return not statically_known_true(math.prod(shape) < 16)
    return True


def tensor_product(module, x, y, weight):
    """Expose exact channel-polynomial adjoints to CPU joint compilation."""
    from ..polynomial import native_polynomial

    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
    count = math.prod(shape)
    operands = [
        value.expand(*shape, value.shape[-1]).reshape(count, value.shape[-1])
        for value in (x, y)
    ]
    operands.append(
        weight.reshape(1, weight.shape[-1])
        if module.shared_weights
        else weight.expand(*shape, weight.shape[-1]).reshape(count, weight.shape[-1])
    )
    operands.append(x.new_empty((0, module.irreps_out.dim)))
    program = repr((((0, 1, 2, 3), 3, 0, 0),))
    output = native_polynomial(
        "channel", module.kernel_metadata, program, None, *operands
    )[0]
    return output.reshape(*shape, module.irreps_out.dim)


def channel_product(metadata, program, operands):
    """Choose bounded CPU slices or the preceding gather implementation."""
    if (
        cpu_slices(operands[0])
        and minimum_width(metadata) >= config.CHANNEL_SLICES_MIN_WIDTH
    ):
        return slice_product(metadata, program, operands)
    return gathered(metadata, program, operands)


@lru_cache(maxsize=128)
def contraction_groups(metadata, program):
    """Collect compatible monomials without expanding tensor operands."""
    _, rows = channel_product_layout(metadata)
    groups = defaultdict(list)
    for mapping, output, slot, required in parse_metadata(program):
        for indices, coefficient, width, strides in rows:
            if indices[output] < 0 or any(
                index < 0
                for role, index in enumerate(indices)
                if required & (1 << role)
            ):
                continue
            roles = tuple(
                role
                for role, index in enumerate(indices)
                if role != output and index >= 0
            )
            key = slot, mapping[output], tuple(mapping[role] for role in roles)
            for channel in range(width):
                groups[key].append(
                    (
                        indices[output] + channel * strides[output],
                        coefficient,
                        tuple(
                            indices[role] + channel * strides[role] for role in roles
                        ),
                    )
                )
    return tuple((key, tuple(values)) for key, values in groups.items())


def empty_outputs(metadata, program, operands):
    """Allocate connected zeros, including disconnected or empty polynomials."""
    dimensions, _ = channel_product_layout(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (1 + max(slot for _, _, slot, _ in terms))
    zero = sum(value[..., :0].sum() for value in operands)
    for mapping, role, slot, _ in terms:
        rows = (
            1
            if statically_known_true(operands[mapping[role]].shape[0] == 1)
            else operands[0].shape[0]
        )
        if outputs[slot] is None:
            outputs[slot] = operands[0].new_zeros((rows, dimensions[role])) + zero
    return outputs


def gathered(metadata, program, operands):
    """Evaluate grouped products and shared-operand reductions with Torch."""
    outputs = empty_outputs(metadata, program, operands)
    nodes, device = operands[0].shape[0], operands[0].device
    for (slot, output, factors), entries in contraction_groups(metadata, program):
        indices, coefficients, columns = zip(*entries)
        value = operands[0].new_tensor(coefficients).unsqueeze(0)
        for role, factor in enumerate(factors):
            index = torch.tensor([column[role] for column in columns], device=device)
            value = value * operands[factor].index_select(1, index)
        value = value.expand(nodes, -1)
        if statically_known_true(operands[output].shape[0] == 1):
            value = value.sum(0, keepdim=True)
        index = torch.tensor(indices, device=device)
        outputs[slot] = outputs[slot].index_add(1, index, value)
    return outputs


def reference(metadata, program, operands):
    """Evaluate each coefficient and channel range independently."""
    _, rows = channel_product_layout(metadata)
    outputs = empty_outputs(metadata, program, operands)
    nodes, device = operands[0].shape[0], operands[0].device
    for mapping, output, slot, required in parse_metadata(program):
        for indices, coefficient, width, strides in rows:
            if indices[output] < 0 or any(
                index < 0
                for role, index in enumerate(indices)
                if required & (1 << role)
            ):
                continue
            channels = torch.arange(width, device=device)
            value = operands[0].new_full((1, width), coefficient)
            for role, index in enumerate(indices):
                if role != output and index >= 0:
                    value = value * operands[mapping[role]].index_select(
                        1, index + channels * strides[role]
                    )
            value = value.expand(nodes, -1)
            if statically_known_true(operands[mapping[output]].shape[0] == 1):
                value = value.sum(0, keepdim=True)
            outputs[slot] = outputs[slot].index_add(
                1, indices[output] + channels * strides[output], value
            )
    return outputs


@lru_cache(maxsize=256)
def slice_plan(metadata, program):
    """Preserve fixed channel runs and output packing across transposes."""
    dimensions, rows = channel_product_layout(metadata)
    slots = defaultdict(lambda: defaultdict(list))
    output_roles = {}
    for mapping, output, slot, required in parse_metadata(program):
        output_roles[slot] = mapping[output], dimensions[output]
        for indices, coefficient, width, strides in rows:
            if indices[output] < 0 or any(
                index < 0
                for role, index in enumerate(indices)
                if required & (1 << role)
            ):
                continue
            factors = tuple(
                (mapping[role], index, strides[role])
                for role, index in enumerate(indices)
                if role != output and index >= 0
            )
            key = indices[output], width if strides[output] else 1, strides[output] or 1
            slots[slot][key].append((width, strides[output] == 0, factors, coefficient))
    result = []
    for slot, (operand, size) in sorted(output_roles.items()):
        groups = tuple(sorted(slots[slot].items()))
        columns = [
            offset + j * stride
            for (offset, width, stride), _ in groups
            for j in range(width)
        ]
        if len(columns) != len(set(columns)):
            return None
        position = {column: index for index, column in enumerate(columns)}
        order = tuple(position.get(column, -1) for column in range(size))
        blocks = []
        index = 0
        while index < len(groups):
            (offset, width, stride), _ = groups[index]
            if stride == 1:
                blocks.append((index, index + 1, False))
                index += 1
                continue
            end = index + stride
            if end <= len(groups) and all(
                groups[index + j][0] == (offset + j, width, stride)
                for j in range(stride)
            ):
                blocks.append((index, end, True))
                index = end
            else:
                blocks = None
                break
        if blocks is not None:
            expanded = []
            for start, stop, interleave in blocks:
                if interleave:
                    expanded.extend(
                        groups[i][0][0] + j * groups[i][0][2]
                        for j in range(groups[start][0][1])
                        for i in range(start, stop)
                    )
                else:
                    offset, width, stride = groups[start][0]
                    expanded.extend(offset + j * stride for j in range(width))
            if expanded != list(range(size)):
                blocks = None
        result.append((slot, operand, size, groups, order, blocks))
    return tuple(result)


def slice_product(metadata, program, operands):
    """Evaluate compact slices and shared reductions with ordinary Torch."""
    specification = slice_plan(metadata, program)
    if specification is None:
        return gathered(metadata, program, operands)
    nodes = operands[0].shape[0]
    outputs = empty_outputs(metadata, program, operands)
    slices = {}
    products = {}
    zero = sum(value[..., :0].sum() for value in operands)
    for slot, output, size, groups, order, blocks in specification:
        values = []
        shared = statically_known_true(operands[output].shape[0] == 1)
        for _, terms in groups:
            value = None
            for width, reduce_channels, factors, coefficient in terms:
                signature = width, factors
                if signature not in products:
                    product = None
                    for factor, offset, stride in factors:
                        key = factor, offset, width, stride
                        if key not in slices:
                            slices[key] = (
                                operands[factor][:, offset : offset + 1]
                                if stride == 0
                                else operands[factor][
                                    :, offset : offset + width * stride : stride
                                ]
                            )
                        product = (
                            slices[key] if product is None else product * slices[key]
                        )
                    products[signature] = (
                        operands[0].new_ones((1, width)) if product is None else product
                    )
                term = products[signature] * coefficient
                if reduce_channels:
                    term = term.expand(-1, width).sum(-1, keepdim=True)
                value = term if value is None else value + term
            value = value.expand(nodes, -1)
            if shared:
                value = value.sum(0, keepdim=True)
            values.append(value)
        if not values:
            continue
        if blocks is not None:
            parts = [
                torch.stack(values[start:stop], -1).flatten(-2)
                if interleave
                else values[start]
                for start, stop, interleave in blocks
            ]
            packed = torch.cat(parts, -1) if len(parts) > 1 else parts[0]
        else:
            joined = torch.cat(values, -1)
            if -1 in order:
                joined = torch.cat((joined, torch.zeros_like(joined[:, :1])), -1)
                order = tuple(joined.shape[1] - 1 if i < 0 else i for i in order)
            index = torch.tensor(order, dtype=torch.long, device=joined.device)
            packed = joined.index_select(1, index)
        outputs[slot] = packed + zero
    return outputs
