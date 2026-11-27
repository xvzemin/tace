"""Native reciprocal radial rows and exact recursively generated adjoints."""

import operator
import os
import re
from functools import lru_cache

import torch

from equivariantx.ops.common.contraction import adjoint_program
from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs


@lru_cache(maxsize=256)
def compressed_source(source, program, output_only=False):
    """Redirect radial reads and reduce repeated-row adjoints inside CG kernels."""
    input_roles, output_roles = {}, {}
    for mapping, _, pairs in program:
        for role, pointer in enumerate(mapping):
            input_roles.setdefault(pointer, set()).add(role)
        for role, slot in pairs:
            output_roles.setdefault(slot, set()).add(role)
    if any(
        1 in roles and roles != {1}
        for roles in (*input_roles.values(), *output_roles.values())
    ):
        raise ValueError("Radial and nonradial buffers cannot alias")

    def address(match):
        kind, slot, expr = match["kind"], int(match["slot"]), match["expr"]
        roles = (input_roles if kind == "p" else output_roles).get(slot, set())
        if roles == {1} and (kind == "g" or not output_only):
            expr = re.sub(r"\bedge\b", "radial_index[edge]", expr)
        return f"{kind}{slot}[{expr}]"

    source = re.sub(
        r"(?P<kind>[pg])(?P<slot>\d+)\[(?P<expr>[^\]\n]+)\]", address, source
    )
    lines = []
    for line in source.splitlines():
        match = re.fullmatch(
            r"if \(active\) (g(\d+)\[.+\]) (?:\+=|=) (\w+);", line.strip()
        )
        if match and output_roles.get(int(match[2])) == {1}:
            line = f"if (active) atomicAdd(&{match[1]}, {match[3]});"
        pointer_store = re.fullmatch(
            r"if \(active\) \*\((g(\d+) [^;]+)\) = (\w+);", line.strip()
        )
        if pointer_store and output_roles.get(int(pointer_store[2])) == {1}:
            address = re.sub(r"\bedge\b", "radial_index[edge]", pointer_store[1])
            line = f"if (active) atomicAdd({address}, {pointer_store[3]});"
        lines.append(line)
    source = "\n".join(lines)
    source = re.sub(
        r"(void (?:run|phase\d+)\([^)]*)(\)\s*\{)",
        r"\1, const int64_t* radial_index\2",
        source,
    )
    source = re.sub(
        r"(case \d+: phase\d+\([^;]+)(\); break;)", r"\1, radial_index\2", source
    )
    return source


def paired_outputs(program, operands, rows):
    """Allocate radial adjoints on pair rows and other results on their inputs."""
    terms = parse_metadata(program)
    outputs = [None] * (1 + max(slot for _, _, pairs in terms for _, slot in pairs))
    for mapping, _, pairs in terms:
        for role, slot in pairs:
            if outputs[slot] is None:
                value = operands[mapping[role]]
                shape = (rows, value.shape[1]) if role == 1 else value.shape
                outputs[slot] = value.new_empty(shape)
    return outputs


def run_paired(metadata, program, source, target, operands, radial_map, rows):
    """Keep directed inputs and reduce radial adjoints within the native kernel."""
    from .edge import complete_weight_columns, run_edges, scheduled_edges
    from .packed_edge import packed_description

    meta, terms = parse_metadata(metadata), parse_metadata(program)
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    paths = meta[0]
    supported = (
        (len(meta) == 3 or packed_description(metadata) is not None)
        and not meta[2]
        and paths
        and all(p[4] == 1 and p[8] >= 0 for p in paths)
        and len({p[3] for p in paths}) == 1
        and all(
            not operands[m[2]].numel() and operands[m[1]].shape[0] != 1
            for m, _, _ in terms
        )
    )
    if supported and roles in ({0, 1, 3}, {1, 3}):
        return run_edges(
            metadata,
            program,
            source,
            target,
            operands,
            edge_block=True,
            radial_map=radial_map,
            radial_rows=rows,
        )
    owner = os.environ.get("EQX_XL_MIXED_OWNER", "off")
    if (
        supported
        and roles == {0, 1}
        and owner in ("source", "target")
        and paths[0][3] == 128
        and len(paths) > 5
        and all(
            complete_weight_columns(metadata, operands[m[1]].shape[1])
            for m, _, _ in terms
        )
    ):
        from .source_vjp import source_vjp_edges

        return source_vjp_edges(
            metadata,
            program,
            source,
            target,
            operands,
            grouping="input",
            full_row=True,
            owner=int(owner == "target"),
            radial_map=radial_map,
            radial_rows=rows,
        )
    outputs = scheduled_edges(metadata, program, source, target, operands)
    slots = {slot for _, _, pairs in terms for role, slot in pairs if role == 1}
    for slot in slots:
        value = outputs[slot]
        outputs[slot] = value.new_zeros((rows, value.shape[1])).index_add_(
            0, radial_map, value
        )
    return outputs


@torch.library.custom_op(
    "eqx_oam_xl::paired_edge", mutates_args=(), device_types="cuda"
)
def paired_edge(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
    radial_map: torch.Tensor,
    rows: int,
) -> list[torch.Tensor]:
    """Reduce explicit, already differentiated radial results into pair rows."""
    return run_paired(metadata, program, source, target, operands, radial_map, rows)


@paired_edge.register_fake
def fake_paired(metadata, program, source, target, operands, radial_map, rows):
    return paired_outputs(program, operands, rows)


def pair_radial_gradients(graph):
    """Fuse single-use radial index reductions after differentiating the loss."""
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    from .program import refresh_inputs

    count = 0
    order = {node: i for i, node in enumerate(graph.nodes)}
    for edge in list(graph.nodes):
        parallel = str(edge.target) == "eqx_oam_xl.parallel_edge_mm.default"
        if not parallel and str(edge.target) != "eqx_oam_xl.edge.default":
            continue
        slots = {}
        for _, _, pairs in parse_metadata(edge.args[1]):
            for role, slot in pairs:
                slots.setdefault(slot, set()).add(role)
        if any(1 in roles and roles != {1} for roles in slots.values()):
            continue
        radial = {slot for slot, roles in slots.items() if roles == {1}}
        if not radial:
            continue
        if parallel:
            collections = [
                node
                for node in edge.users
                if node.target == operator.getitem and node.args[1] == 0
            ]
            if len(collections) != 1:
                continue
            collection = collections[0]
        else:
            collection = edge
        if any(node.target != operator.getitem for node in collection.users):
            continue
        items = {
            node.args[1]: node
            for node in collection.users
            if node.target == operator.getitem
        }
        if not radial <= items.keys():
            continue
        if any(
            sum(node.args[1] == slot for node in collection.users) != 1
            for slot in radial
        ):
            continue
        matches = []
        for slot in sorted(radial):
            item = items[slot]
            consumers = [
                node
                for node in item.users
                if node.target != torch.ops.aten.new_zeros.default
            ]
            if len(consumers) != 1:
                break
            (reduction,) = consumers
            if (
                reduction.target != torch.ops.aten.index_put.default
                or reduction.args[2] is not item
            ):
                break
            base, indices = reduction.args[:2]
            if (
                len(indices) != 1
                or indices[0] is None
                or order[indices[0]] >= order[edge]
                or base.target
                not in (torch.ops.aten.new_zeros.default, torch.ops.aten.zeros.default)
                or not (
                    reduction.args[3]
                    if len(reduction.args) > 3
                    else reduction.kwargs.get("accumulate", False)
                )
                or any(
                    user is not reduction
                    and (user is not base or set(user.users) != {reduction})
                    for user in item.users
                )
            ):
                break
            matches.append((slot, item, reduction, indices[0]))
        if len(matches) != len(radial) or len({item[3] for item in matches}) != 1:
            continue
        expected = matches[0][2].meta["val"].shape[0]
        shapes = [
            node
            for node in graph.nodes
            if node.op == "placeholder"
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].ndim == 1
            and statically_known_true(node.meta["val"].shape[0] == expected)
        ]
        if not shapes or not all(
            statically_known_true(item[2].meta["val"].shape[0] == expected)
            for item in matches
        ):
            continue
        with graph.inserting_before(edge):
            rows = graph.call_function(torch.ops.aten.sym_size.int, (shapes[0], 0))
            rows.meta["val"] = expected
        edge.target = (
            torch.ops.eqx_oam_xl.parallel_paired_mm.default
            if parallel
            else torch.ops.eqx_oam_xl.paired_edge.default
        )
        edge.args = (*edge.args, matches[0][3], rows)
        values = list(edge.meta["val"][0] if parallel else edge.meta["val"])
        for slot, item, reduction, _ in matches:
            values[slot] = reduction.meta["val"]
            item.meta["val"] = values[slot]
            if "example_value" in item.meta:
                item.meta["example_value"] = values[slot]
            reduction.replace_all_uses_with(item)
            graph.erase_node(reduction)
        edge.meta["val"] = (values, edge.meta["val"][1]) if parallel else values
        collection.meta["val"] = values
        for node in {edge, collection}:
            if "example_value" in node.meta:
                node.meta["example_value"] = node.meta["val"]
        refresh_inputs(edge)
        count += 1
    graph.eliminate_dead_code()
    graph.lint()
    return count


def run_compressed(metadata, program, source, target, operands, radial_map):
    """Evaluate every explicit term without expanding the radial edge arrays."""
    from .cg_convolution import contract_direct

    terms = parse_metadata(program)
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    if os.environ.get("EQX_XL_COMPRESSED_SCHEDULED", "1") == "1":
        from .edge import RECEIVER_PIPELINE, run_edges

        if source.numel() >= 1024 and roles in ({0, 1, 3}, {1, 3}):
            return run_edges(
                metadata, program, source, target, operands, radial_map=radial_map
            )
        paths = parse_metadata(metadata)[0]
        owner = os.environ.get("EQX_XL_MIXED_OWNER", "off")
        if (
            roles == {0, 1}
            and owner in ("source", "target")
            and len(paths) > 5
            and all(p[3] == 128 and p[4] == 1 and p[8] >= 0 for p in paths)
        ):
            from .source_vjp import source_vjp_edges

            return source_vjp_edges(
                metadata,
                program,
                source,
                target,
                operands,
                grouping="input",
                full_row=True,
                owner=int(owner == "target"),
                radial_map=radial_map,
            )
        if (
            RECEIVER_PIPELINE != "off"
            and roles == {4}
            and paths
            and all(p[3] == 128 and p[4] == 1 and p[8] >= 0 for p in paths)
        ):
            from .receiver import receiver_edges

            asynchronous = RECEIVER_PIPELINE == "async" or (
                RECEIVER_PIPELINE == "auto" and len(terms) == 1
            )
            return receiver_edges(
                metadata, program, source, target, operands, asynchronous, radial_map
            )
    if (
        os.environ.get("EQX_XL_RADIAL_OWNER", "0") == "1"
        and 1 in roles
        and all(operands[mapping[1]].shape[0] != 1 for mapping, _, _ in terms)
    ):
        from .radial_owner import radial_owned_edges

        return radial_owned_edges(
            metadata, program, source, target, operands, radial_map
        )
    outputs = o3_outputs(metadata, program, source, target, operands)
    for output in outputs:
        output.zero_()
    if not source.numel():
        return outputs
    required = {
        index
        for mapping, _, pairs in terms
        for output, _ in pairs
        for role, index in enumerate(mapping)
        if role != output
    }
    operands = [x.contiguous() if i in required else x for i, x in enumerate(operands)]
    if any(operands[mapping[2]].numel() for mapping, _, _ in terms):
        raise ValueError("Compressed radial heads must be materialized")
    calls = [
        (
            tuple(role for role, _ in pairs),
            tuple(operands[i] for i in mapping),
            tuple(outputs[slot] for _, slot in pairs),
            weighted,
        )
        for mapping, weighted, pairs in terms
    ]
    contract_direct(
        metadata,
        source.contiguous(),
        target.contiguous(),
        calls,
        batch_paths=True,
        radial_map=radial_map.contiguous(),
    )
    return outputs


@torch.library.custom_op(
    "eqx_oam_xl::compressed_edge", mutates_args=(), device_types="cuda"
)
def compressed_edge(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
    radial_map: torch.Tensor,
) -> list[torch.Tensor]:
    return run_compressed(metadata, program, source, target, operands, radial_map)


@compressed_edge.register_fake
def fake_compressed(metadata, program, source, target, operands, radial_map):
    return o3_outputs(metadata, program, source, target, operands)


def compressed_context(ctx, inputs, output):
    metadata, program, source, target, operands, radial_map = inputs
    ctx.kernel_metadata, ctx.program = metadata, parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, radial_map, *operands)


def compressed_backward(ctx, grad_outputs):
    source, target, radial_map, *operands = ctx.saved_tensors
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        gradient_mask(operands, ctx.needs_input_grad[4]),
        parse_metadata(ctx.kernel_metadata)[2],
    )
    gradients = [None] * len(operands)
    if program:
        outputs = compressed_edge(
            ctx.kernel_metadata, repr(program), source, target, values, radial_map
        )
        for index, slot in destinations.items():
            gradients[index] = outputs[slot]
    return None, None, None, None, gradients, None


compressed_edge.register_autograd(compressed_backward, setup_context=compressed_context)
