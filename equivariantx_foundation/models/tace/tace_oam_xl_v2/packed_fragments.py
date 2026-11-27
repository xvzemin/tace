"""Pass irrep cotangent blocks directly to explicit node-product programs.

The transformation runs after differentiation. It replaces only a contiguous
flat concatenation whose pieces exactly match the packed irrep layout. Every
piece remains a live graph input; neither values nor model parameters are
cached. Forward and higher derivative formulas are already explicit here.
"""

from functools import lru_cache

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from equivariantx.utils.metadata import parse_metadata


@lru_cache(maxsize=256)
def fragment_source(source, metadata, fragments):
    """Redirect packed reads without changing arithmetic or output addresses."""
    width, _, blocks, _ = parse_metadata(metadata)
    for slot, parts in fragments:
        if len(parts) != len(blocks):
            raise ValueError("One fragment is required per packed irrep block")
        begin = 0
        for part, size in zip(parts, blocks):
            source = source.replace(f"p{slot}[nodes*{begin * width}+", f"p{part}[")
            begin += size
        if f"p{slot}[" in source:
            raise ValueError("A fragmented slot still has an unhandled read")
    return source


def validate_fragments(metadata, program, operands, fragments):
    """Reject aliases to other operand roles and incompatible block layouts."""
    width, _, blocks, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    nodes = operands[0].shape[0]
    seen = set()
    for slot, parts in fragments:
        if slot <= 0 or slot >= len(operands) or slot in seen:
            raise ValueError("Fragmented operands must have unique nonzero slots")
        seen.add(slot)
        if len(parts) != len(blocks):
            raise ValueError("Fragment count does not match the packed layout")
        if any(
            mapping[role] == slot
            for mapping, output, _, _ in terms
            for role in range(3)
            if role != output
        ):
            raise ValueError("A fragmented slot is also read in an unpacked role")
        for part, size in zip(parts, blocks):
            if part <= slot or part >= len(operands):
                raise ValueError(
                    "Fragments must be appended after the original operand"
                )
            value = operands[part]
            if (
                value.ndim != 2
                or not value.is_contiguous()
                or value.numel() != nodes * size * width
                or value.dtype != operands[0].dtype
                or value.device != operands[0].device
            ):
                raise ValueError(
                    "A fragment has an incompatible shape, dtype, or device"
                )


@torch.library.custom_op(
    "eqx_oam_xl::fragmented_product", mutates_args=(), device_types="cuda"
)
def fragmented_product(
    metadata: str, program: str, operands: list[torch.Tensor], fragments: str
) -> list[torch.Tensor]:
    """Evaluate a differentiated product from live, unjoined cotangent blocks."""
    from .packed_product import run_packed

    plan = parse_metadata(fragments)
    validate_fragments(metadata, program, operands, plan)
    return run_packed(metadata, program, operands, fragments=plan)


@fragmented_product.register_fake
def fragmented_outputs(metadata, program, operands, fragments):
    from .packed_product import packed_outputs

    return packed_outputs(metadata, program, operands)


def matching_pieces(value, nodes, width, blocks):
    """Find a provably matching flat cat through contiguous, alias-only views."""
    aliases = {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.alias.default,
    }
    while isinstance(value, torch.fx.Node) and value.target in aliases:
        example = value.meta.get("val")
        if not isinstance(example, torch.Tensor) or not example.is_contiguous():
            return None
        value = value.args[0]
    if (
        not isinstance(value, torch.fx.Node)
        or value.target != torch.ops.aten.cat.default
    ):
        return None
    dim = value.args[1] if len(value.args) > 1 else value.kwargs.get("dim", 0)
    pieces = value.args[0]
    if dim not in (0, -1) or len(pieces) != len(blocks):
        return None
    result = value.meta.get("val")
    if not isinstance(result, torch.Tensor):
        return None
    for piece, size in zip(pieces, blocks):
        example = piece.meta.get("val") if isinstance(piece, torch.fx.Node) else None
        if (
            not isinstance(example, torch.Tensor)
            or example.ndim != 1
            or not example.is_contiguous()
            or example.dtype != result.dtype
            or example.device != result.device
            or not statically_known_true(example.numel() == nodes * size * width)
        ):
            return None
    return pieces


def fragment_packed_graph(graph):
    """Remove full cotangent concatenations at their native product consumers.

    Other consumers retain the original concatenation. Unused placeholder
    arguments carry no dependency on the removed packed tensor. Tensor-list
    layout metadata is refreshed for Inductor's custom-op lowering.
    """
    from .program import refresh_inputs

    count, slots = 0, 0
    for node in list(graph.nodes):
        if str(node.target) != "eqx_oam_xl.packed_product.default":
            continue
        metadata, program, values = node.args
        width, _, blocks, _ = parse_metadata(metadata)
        terms = parse_metadata(program)
        nodes = values[0].meta["val"].shape[0]
        required = {mapping[3] for mapping, role, _, _ in terms if role != 3}
        other_reads = {
            mapping[role]
            for mapping, output, _, _ in terms
            for role in range(3)
            if role != output
        }
        selected = {}
        for slot in sorted(required - other_reads - {0}):
            pieces = matching_pieces(values[slot], nodes, width, blocks)
            if pieces is not None:
                selected[slot] = pieces
        if not selected:
            continue
        values, fragments = list(values), []
        with graph.inserting_before(node):
            for slot, pieces in selected.items():
                original = values[slot].meta["val"]
                placeholder = graph.call_function(
                    torch.ops.aten.empty.memory_format,
                    ([0, 0],),
                    {
                        "dtype": original.dtype,
                        "device": original.device,
                        "pin_memory": False,
                    },
                )
                placeholder.meta["val"] = original.new_empty((0, 0))
                values[slot] = placeholder
                parts = []
                for piece in pieces:
                    flat = graph.call_function(
                        torch.ops.aten.view.default, (piece, [1, -1])
                    )
                    flat.meta["val"] = piece.meta["val"].view(1, -1)
                    parts.append(len(values))
                    values.append(flat)
                fragments.append((slot, tuple(parts)))
        node.target = torch.ops.eqx_oam_xl.fragmented_product.default
        node.args = (metadata, program, values, repr(tuple(fragments)))
        refresh_inputs(node)
        count += 1
        slots += len(fragments)
    graph.eliminate_dead_code()
    graph.lint()
    return {"products": count, "operands": slots}
