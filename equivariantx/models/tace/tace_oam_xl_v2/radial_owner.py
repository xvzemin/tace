"""Reduce repeated radial adjoints on their unique representative rows."""

from functools import lru_cache
import re
import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from .graph import prepare_graph
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .edge_codegen import edge_groups, edge_source
from .packed_edge import packed_description, pack_edge_source
from .compressed import compressed_source


def radial_source(source, program):
    """Move radial accumulation around the one-or-more-edge representative row."""
    weight_slots = {
        slot for _, _, pairs in program for role, slot in pairs if role == 1
    }
    declarations, stores, body = [], [], []
    sum_names = set()
    pattern = re.compile(
        r"if \(active\) \*\((g(\d+) \+ edge \* \d+ \+ \d+) \+ u\) = (sum\d+);"
    )
    for line in source.splitlines():
        match = pattern.fullmatch(line.strip())
        if match and int(match[2]) in weight_slots:
            sum_names.add(match[3])
            stores.append(
                f"if (active) *({match[1].replace('edge', 'row')} + u) = {match[3]};"
            )
        else:
            body.append(line)
    if not sum_names:
        raise ValueError("No exclusive radial store found")
    kept = []
    for line in body:
        if any(line.strip() == f"T {name} = 0;" for name in sum_names):
            declarations.append(line)
        else:
            kept.append(line)
    source = "\n".join(kept)
    marker = "const int64_t edge = row;"
    if source.count(marker) != 1:
        raise ValueError("Expected one edge traversal")
    source = source.replace(
        marker,
        "\n".join(declarations)
        + "\nfor (int64_t at=pointers[row];at<pointers[row+1];++at) {\nconst int64_t edge=order[at];",
    )
    end = source.rfind("}")
    return source[:end] + "}\n" + "\n".join(stores) + "\n}"


@lru_cache(maxsize=256)
def owned_plan(metadata, program, dimensions, shared, dtype, device):
    terms = parse_metadata(program)
    paths = parse_metadata(metadata)[0]
    outputs = 1 + max(slot for _, _, pairs in terms for _, slot in pairs)
    groups = edge_groups(paths, terms)
    sources = []
    for paths, terms, owner in groups:
        code = edge_source(paths, terms, dimensions, shared, dtype, outputs, owner)
        if owner < 0:
            code = radial_source(code, terms)
        packing = packed_description(metadata)
        if packing is not None:
            code = pack_edge_source(code, packing, terms)
        code = compressed_source(code, terms)
        sources.append(code)
    compiled = kernels(sources, device)
    return tuple(
        (compiled[code], paths[0][3], owner)
        for code, (paths, _, owner) in zip(sources, groups)
    )


def radial_owned_edges(metadata, program, source, target, operands, radial_map):
    """Use complete CSR rows for nodes and for the compressed radial adjoints."""
    terms = parse_metadata(program)
    required = {
        index
        for mapping, _, pairs in terms
        for output, _ in pairs
        for role, index in enumerate(mapping)
        if role != output
    }
    operands = [x.contiguous() if i in required else x for i, x in enumerate(operands)]
    source, target, radial_map = (
        source.contiguous(),
        target.contiguous(),
        radial_map.contiguous(),
    )
    outputs = o3_outputs(metadata, program, source, target, operands)
    for output in outputs:
        output.zero_()
    if not source.numel():
        return outputs
    stages = owned_plan(
        metadata,
        program,
        tuple(x.shape[1] for x in operands),
        tuple(x.shape[0] == 1 for x in operands),
        "float" if operands[0].dtype == torch.float32 else "double",
        operands[0].device,
    )
    mapping = terms[0][0]
    counts = {
        0: operands[mapping[0]].shape[0],
        4: operands[mapping[4]].shape[0],
        -1: operands[mapping[1]].shape[0],
    }
    orders = {
        owner: prepare_graph(radial_map, radial_map, 0, counts[-1])
        if owner < 0
        else prepare_graph(source, target, int(owner == 4), counts[owner])
        for owner in {owner for _, _, owner in stages}
    }
    pointer_values = [x.data_ptr() for x in (*operands, *outputs)]
    extra = ([counts[0]] if packed_description(metadata) is not None else []) + [
        radial_map.data_ptr()
    ]
    launches = []
    for kernel, width, owner in stages:
        order, offsets = orders[owner]
        rows = counts[owner]
        arguments = (
            pointer_values
            + [
                source.data_ptr(),
                target.data_ptr(),
                order.data_ptr(),
                offsets.data_ptr(),
                rows,
            ]
            + extra
        )
        threads = max(
            (64, 128, 256),
            key=lambda n: (n * kernel.active_blocks(n, 0), -abs(n - 128)),
        )
        launches.append(
            (
                kernel,
                arguments,
                (rows + threads // 32 - 1) // (threads // 32),
                (width + 31) // 32,
                threads,
                0,
            )
        )
    runtime().launch(
        launches, torch.cuda.current_stream(operands[0].device).cuda_stream
    )
    return outputs
