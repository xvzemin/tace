"""Batched contractions with independent instruction weights."""

import math
from collections import defaultdict
from typing import NamedTuple

import torch


class InstructionGroup(NamedTuple):
    coupling: tuple
    connection_mode: str
    path_shape: tuple
    i_in1: tuple
    i_in2: tuple
    i_out: tuple
    weight_slices: tuple
    weight_slice: slice | None
    scale_slice: slice
    path_weights: tuple


def instruction_groups(instructions, couplings):
    """Group compatible contractions without changing instruction indices."""
    entries = defaultdict(list)
    offset = 0
    for index, (ins, coupling) in enumerate(zip(instructions, couplings)):
        size = math.prod(ins.path_shape) if ins.has_weight else 0
        key = (coupling, ins.connection_mode, ins.has_weight, ins.path_shape)
        entries[key].append((index, slice(offset, offset + size)))
        offset += size
    groups, order = [], []
    for (coupling, mode, weighted, path_shape), paths in entries.items():
        selected = [instructions[i] for i, _ in paths]
        inputs = [tuple(p.i_in1 for p in selected), tuple(p.i_in2 for p in selected)]
        inputs = [ids[:1] if len(set(ids)) == 1 else ids for ids in inputs]
        sections = tuple(section for _, section in paths) if weighted else ()
        contiguous = sections and all(
            a.stop == b.start for a, b in zip(sections, sections[1:])
        )
        groups.append(
            InstructionGroup(
                coupling,
                mode,
                path_shape,
                *inputs,
                tuple(p.i_out for p in selected),
                sections,
                slice(sections[0].start, sections[-1].stop) if contiguous else None,
                slice(len(order), len(order) + len(paths)),
                tuple(p.path_weight for p in selected),
            )
        )
        order.extend(i for i, _ in paths)
    return tuple(order), tuple(groups)


class TensorProduct(torch.nn.Module):
    """Evaluate instruction batches using fixed feature and weight indices.

    Parameters
    ----------
    shapes : sequence of sequence of tuple
        (Multiplicity, angular dimension) entries for both inputs and output.
    groups : sequence of InstructionGroup
        Compatible instruction batches.
    couple : callable
        Angular contraction on broadcast (..., channels, dimension) inputs.
    layout : {"mul_ir", "ir_mul"}, optional
        Input and output feature layout.
    """

    def __init__(self, shapes, groups, couple, layout="mul_ir"):
        super().__init__()
        self.groups, self.couple, self.layout = groups, couple, layout
        self.shapes = tuple(tuple(entries) for entries in shapes)
        self.indexed = layout == "ir_mul" and len(groups) * 2 < sum(
            len(group.i_out) for group in groups
        )
        offsets = []
        for entries in shapes:
            starts, offset = [], 0
            for mul, dim in entries:
                starts.append(offset)
                offset += mul * dim
            offsets.append(starts)
        self.output_dim = sum(mul * dim for mul, dim in shapes[2])
        self.input_shapes = []
        self.input_sections = []
        output_indices = []
        for p, group in enumerate(groups):
            sizes = []
            sections = []
            for role, entries in enumerate((group.i_in1, group.i_in2)):
                indices = []
                for i in entries:
                    mul, dim = shapes[role][i]
                    indices.extend(
                        offsets[role][i]
                        + (u * dim + m if layout == "mul_ir" else m * mul + u)
                        for u in range(mul)
                        for m in range(dim)
                    )
                sizes.append((len(entries), *shapes[role][entries[0]]))
                starts = [offsets[role][i] for i in entries]
                width = math.prod(shapes[role][entries[0]])
                sections.append(
                    slice(starts[0], starts[0] + len(entries) * width)
                    if starts
                    == list(
                        range(starts[0], starts[0] + len(entries) * width, width or 1)
                    )
                    else None
                )
                if self.indexed and sections[-1] is None:
                    self.register_buffer(
                        f"input_{role}_{p}",
                        torch.tensor(indices, dtype=torch.long),
                        persistent=False,
                    )
            self.input_shapes.append(tuple(sizes))
            self.input_sections.append(tuple(sections))
            indices = [
                i
                for section in group.weight_slices
                for i in range(section.start, section.stop)
            ]
            if self.indexed and group.weight_slices and group.weight_slice is None:
                self.register_buffer(
                    f"weight_{p}",
                    torch.tensor(indices, dtype=torch.long),
                    persistent=False,
                )
            for i in group.i_out:
                mul, dim = shapes[2][i]
                output_indices.extend(
                    offsets[2][i] + (u * dim + m if layout == "mul_ir" else m * mul + u)
                    for u in range(mul)
                    for m in range(dim)
                )
        self.input_shapes = tuple(self.input_shapes)
        self.input_sections = tuple(self.input_sections)
        sections = sorted(
            {(s.start, s.stop) for group in groups for s in group.weight_slices}
        )
        self.weight_sizes = tuple(stop - start for start, stop in sections)
        section_indices = {section: i for i, section in enumerate(sections)}
        self.weight_indices = tuple(
            tuple(section_indices[s.start, s.stop] for s in group.weight_slices)
            for group in groups
        )
        self.permutation = (
            len(output_indices) == self.output_dim
            and len(set(output_indices)) == self.output_dim
        )
        self.identity = output_indices == list(range(self.output_dim))
        self.complete = len(set(output_indices)) == self.output_dim
        if self.indexed:
            index = torch.tensor(output_indices, dtype=torch.long)
            self.register_buffer(
                "output_index",
                index.argsort() if self.permutation else index,
                persistent=False,
            )

    def forward(self, x, y, weight, scales):
        """Return flattened features without merging independent path weights."""
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        outputs = []
        if not self.indexed:
            weights = weight.split(self.weight_sizes, -1)
            inputs = [
                [
                    (
                        block.reshape(*value.shape[:-1], mul, dim)
                        if self.layout == "mul_ir"
                        else block.reshape(*value.shape[:-1], dim, mul).transpose(
                            -1, -2
                        )
                    )
                    for block, (mul, dim) in zip(
                        value.split([mul * dim for mul, dim in entries], -1), entries
                    )
                ]
                for value, entries in zip((x, y), self.shapes[:2])
            ]
            outputs = [None] * len(self.shapes[2])
        for p, group in enumerate(self.groups):
            operands = []
            single = not self.indexed and len(group.i_out) == 1
            for role, (value, dims, section, indices) in enumerate(
                zip(
                    (x, y),
                    self.input_shapes[p],
                    self.input_sections[p],
                    (group.i_in1, group.i_in2),
                )
            ):
                if single:
                    operand = inputs[role][indices[0]]
                elif not self.indexed:
                    operand = (
                        inputs[role][indices[0]].unsqueeze(-3)
                        if len(indices) == 1
                        else torch.stack([inputs[role][i] for i in indices], -3)
                    )
                elif section is None:
                    operand = value.index_select(
                        -1, getattr(self, f"input_{role}_{p}")
                    ).reshape(*value.shape[:-1], *dims)
                else:
                    operand = (
                        value[..., section]
                        .reshape(*value.shape[:-1], dims[0], dims[2], dims[1])
                        .transpose(-1, -2)
                    )
                operands.append(operand)
            left, right = operands
            mode = group.connection_mode
            if mode not in ("u1u", "uuu", "uuw"):
                left, right = left.unsqueeze(-2), right.unsqueeze(-3)
            value = self.couple(left, right, *group.coupling)
            scale = group.path_weights[0] if single else scales[group.scale_slice]
            if group.weight_slices:
                if not self.indexed:
                    indices = self.weight_indices[p]
                    w = (
                        weights[indices[0]]
                        if len(indices) == 1
                        else torch.cat([weights[i] for i in indices], -1)
                    )
                else:
                    w = (
                        weight[..., group.weight_slice]
                        if group.weight_slice is not None
                        else weight.index_select(-1, getattr(self, f"weight_{p}"))
                    )
                w = w.reshape(
                    *weight.shape[:-1],
                    *((() if single else (len(group.i_out),)) + group.path_shape),
                )
                if mode != "uvw":
                    w = w * (
                        scale
                        if single
                        else scale.reshape(-1, *([1] * len(group.path_shape)))
                    )
                if mode == "uvw":
                    value = torch.einsum("...uvd,...uvw->...wd", value, w)
                elif mode == "uuw":
                    value = torch.einsum("...ud,...uw->...wd", value, w)
                else:
                    value = value * w.unsqueeze(-1)
            elif mode == "uuw":
                value = value.sum(-2, keepdim=True)
            if mode == "uvu":
                value = value.sum(-2)
            elif mode == "uvv":
                value = value.sum(-3)
            elif mode == "uvuv":
                value = value.flatten(-3, -2)
            if not group.weight_slices or mode == "uvw":
                value = value * (scale if single else scale[:, None, None])
            if self.indexed:
                outputs.append(value.flatten(-3).expand(*shape, -1))
            else:
                for dest, block in zip(
                    group.i_out, (value,) if single else value.unbind(-3)
                ):
                    outputs[dest] = (
                        block if outputs[dest] is None else outputs[dest] + block
                    )
        zero = (
            x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
            if not self.complete or not outputs
            else None
        )
        if not self.indexed and outputs:
            blocks = [
                x.new_zeros((*shape, mul, dim))
                if value is None
                else value.expand(*shape, mul, dim)
                for value, (mul, dim) in zip(outputs, self.shapes[2])
            ]
            value = torch.cat(
                [
                    block.flatten(-2)
                    if self.layout == "mul_ir"
                    else block.transpose(-1, -2).flatten(-2)
                    for block in blocks
                ],
                -1,
            )
            return value if zero is None else value + zero
        if not outputs:
            return x.new_zeros((*shape, self.output_dim)) + zero
        value = outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)
        if self.identity:
            return value if zero is None else value + zero
        if self.permutation:
            return value.index_select(-1, self.output_index)
        value = x.new_zeros((*shape, self.output_dim)).index_add(
            -1, self.output_index, value
        )
        return value if zero is None else value + zero
