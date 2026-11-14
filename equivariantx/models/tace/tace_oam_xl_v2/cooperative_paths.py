"""Receiver CTAs with one warp per angular path and shared live node inputs.

This experimental schedule preserves every independent UVU radial weight.
Forward/tangent paths own disjoint receiver outputs. Final mixed paths first
reduce source adjoints across their warps, then update the source once per
path tile. Radial adjoints retain one writer per edge/channel.
"""

from collections import defaultdict
from functools import lru_cache
import re

import torch

from equivariantx.ops.common.cuda.codegen import HEADER
from .cg_codegen import angular_source
from .graph import prepare_graph
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .packed_edge import packed_description, pack_edge_source


def path_body(path, terms, dims, shared, mixed):
    """Emit one path's complete value/tangent or final mixed polynomial."""
    cache, body, node_terms, radial_terms = {}, [], defaultdict(list), defaultdict(list)
    for mapping, _, pairs in terms:
        needed = {role for role, _ in pairs}
        values = angular_source(path, mapping, dims, shared, needed, cache, body)
        for role, slot in pairs:
            if role in (0, 4):
                pointer, factor = mapping[1], path[9]
                key = ("radial", pointer, path[8], factor)
                if key not in cache:
                    name = f"v{len(cache)}"
                    body.append(
                        f"const T {name} = T({factor:.17g}) * p{pointer}[edge*{dims[pointer]}+{path[8]}+u];"
                    )
                    cache[key] = name
                for m in range(path[5] if role == 0 else path[7]):
                    node_terms[slot, role, m].append((cache[key], values[0, role, m]))
            elif role == 1:
                radial_terms[slot].append((f"T({path[9]:.17g})", values[0, 1, 0]))
            else:
                raise ValueError(
                    "Cooperative path phase only supports receiver or source/radial results"
                )
    if mixed:
        for (slot, role, m), values in node_terms.items():
            body.append(f"T grad{m}=0;")
            for weight, value in values:
                body.append(f"grad{m}=fma({weight},{value},grad{m});")
            body.append(f"node_partials[(warp*{path[5]}+{m})*32+lane]=grad{m};")
    else:
        for (_, _, m), values in node_terms.items():
            for weight, value in values:
                body.append(f"sum{m}=fma({weight},{value},sum{m});")
    for slot, values in radial_terms.items():
        body.append("{ T radial=0;")
        for weight, value in values:
            body.append(f"radial=fma({weight},{value},radial);")
        width = dims[
            next(mapping[1] for mapping, _, pairs in terms if (1, slot) in pairs)
        ]
        body.append(f"g{slot}[edge*{width}+{path[8]}+u]=radial; }}")
    return "\n".join(body)


def cooperative_source(
    paths, terms, dims, shared, dtype, asynchronous=False, row_tile=1
):
    """Generate a shared-input path tile with an explicit whole-CTA barrier."""
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    mixed = roles == {0, 1}
    if roles not in ({4}, {0, 1}) or len({p[0] for p in paths}) != 1:
        raise ValueError(
            "Cooperative paths need one input irrep and a supported complete phase"
        )
    warps, channels, di = len(paths), paths[0][3], paths[0][5]
    if channels != 128 or any(p[4] != 1 or p[8] < 0 for p in paths):
        raise ValueError("Cooperative paths specialize nonshared 128-channel UVU")
    outputs = 1 + max(slot for _, _, pairs in terms for _, slot in pairs)
    bodies = [path_body(path, terms, dims, shared, mixed) for path in paths]
    expressions = list(
        dict.fromkeys(
            expression
            for body in bodies
            for expression in re.findall(r"p\d+\[[^\]]+\]", body)
        )
    )
    edge_slots, owner_slots, edge_size, owner_size = {}, {}, 0, 0
    for expression in expressions:
        width = 32 if re.search(r"\bu\b", expression) else 1
        if "target_node" in expression:
            owner_slots[expression] = (owner_size, width)
            owner_size += width
        else:
            edge_slots[expression] = (edge_size, width)
            edge_size += width
    edge_size = (edge_size + 31) // 32 * 32
    owner_size = (owner_size + 31) // 32 * 32
    if row_tile > 1 and asynchronous:
        raise ValueError("Batched rows currently use synchronous cooperative staging")
    storage = (
        edge_size * (2 if asynchronous else row_tile)
        + owner_size
        + (row_tile * warps * di * 32 if mixed else 0)
    ) * (4 if dtype == "float" else 8)
    if storage > 48000:
        raise ValueError("Cooperative path shared-memory tile is too large")
    for i, body in enumerate(bodies):
        for expression, (offset, width) in edge_slots.items():
            body = body.replace(
                expression,
                f"edge_values[{'stage' if asynchronous else 'local' if row_tile > 1 else '0'}][{offset}"
                + ("+lane]" if width == 32 else "]"),
            )
        for expression, (offset, width) in owner_slots.items():
            body = body.replace(
                expression,
                f"owner_values[{offset}" + ("+lane]" if width == 32 else "]"),
            )
        bodies[i] = (
            body.replace(
                "node_partials[(warp*", f"node_partials[((local*{warps}+warp)*"
            )
            if row_tile > 1
            else body
        )
    arguments = [f"const T* p{i}" for i in range(len(dims))] + [
        f"T* g{i}" for i in range(outputs)
    ]
    arguments += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* order",
        "const int64_t* pointers",
        "int64_t nodes",
    ]
    code = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ",".join(arguments) + ") {",
        "const int lane=threadIdx.x%32, warp=threadIdx.x/32;",
        "const int u=blockIdx.y*32+lane;",
        "constexpr bool active=true;",
        "const int64_t node=blockIdx.x; if(node>=nodes) return;",
        "const int64_t begin=pointers[node], end=pointers[node+1];",
        f"__shared__ T edge_values[{2 if asynchronous else row_tile}][{max(1, edge_size)}];",
    ]
    if owner_size:
        code += [
            f"__shared__ T owner_values[{owner_size}];",
            "const int64_t target_node=node;",
        ]
        for i, (expression, (offset, width)) in enumerate(owner_slots.items()):
            condition = (
                f"warp=={i % warps}"
                if width == 32
                else f"threadIdx.x=={i % (warps * 32)}"
            )
            address = f"{offset}" + ("+lane" if width == 32 else "")
            code.append(f"if({condition}) owner_values[{address}]={expression};")
        code.append("__syncthreads();")
    if mixed:
        code.append(f"__shared__ T node_partials[{row_tile * warps * di * 32}];")
    else:
        code += [f"T sum{m}=0;" for m in range(max(p[7] for p in paths))]
    if asynchronous:
        code.append("int stage=0;")

    def preload(edge, stage):
        result = ["{", f"const int64_t edge={edge}, source_node=source[edge];"]
        for i, (expression, (offset, width)) in enumerate(edge_slots.items()):
            condition = (
                f"warp=={i % warps}"
                if width == 32
                else f"threadIdx.x=={i % (warps * 32)}"
            )
            address = f"edge_values[{stage}][{offset}" + (
                "+lane]" if width == 32 else "]"
            )
            if asynchronous:
                result.append(
                    f"if({condition}) {{ unsigned int address=static_cast<unsigned int>(__cvta_generic_to_shared(&{address}));"
                )
                result.append(
                    f'asm volatile("cp.async.ca.shared.global [%0], [%1], {4 if dtype == "float" else 8};" :: "r"(address),"l"(&{expression})); }}'
                )
            else:
                result.append(f"if({condition}) {address}={expression};")
        if asynchronous:
            result.append('asm volatile("cp.async.commit_group;");')
        return "\n".join(result + ["}"])

    if row_tile > 1:
        code += [
            f"for(int64_t position=begin;position<end;position+={row_tile}){{",
            f"const int count=min(int64_t({row_tile}),end-position);",
            "for(int local=0;local<count;++local){",
            preload("order[position+local]", "local"),
            "}",
            "__syncthreads();",
            "for(int local=0;local<count;++local){",
            "const int64_t edge=order[position+local], source_node=source[edge];",
        ]
    else:
        if asynchronous:
            code += ["if(begin<end){", preload("order[begin]", "0"), "}"]
        code += [
            "for(int64_t position=begin;position<end;++position){",
            "const int64_t edge=order[position], source_node=source[edge];",
        ]
        if asynchronous:
            code += [
                'asm volatile("cp.async.wait_group 0;");',
                "__syncthreads();",
                "if(position+1<end){",
                preload("order[position+1]", "stage^1"),
                "}",
            ]
        else:
            code += [preload("order[position]", "0"), "__syncthreads();"]
    code.append("switch(warp){")
    code += [f"case {i}:{{{body} break;}}" for i, body in enumerate(bodies)]
    code += ["}"]
    if row_tile > 1:
        code.append("}")
    code.append("__syncthreads();")
    if mixed:
        owners = {
            (slot, mapping[0])
            for mapping, _, pairs in terms
            for role, slot in pairs
            if role == 0
        }
        if len({slot for slot, _ in owners}) != 1:
            raise ValueError("Source result slots must coincide across mixed terms")
        slot, pointer = next(iter(owners))
        code += ["if(warp==0){"]
        if row_tile > 1:
            code += [
                "for(int local=0;local<count;++local){",
                "const int64_t source_node=source[order[position+local]];",
            ]
        for m in range(di):
            group = f"(local*{warps}+p)" if row_tile > 1 else "p"
            code += [
                "{ T total=0;",
                f"for(int p=0;p<{warps};++p) total+=node_partials[({group}*{di}+{m})*32+lane];",
                f"atomicAdd(g{slot}+source_node*{dims[pointer]}+{paths[0][0] + m * 128}+u,total); }}",
            ]
        if row_tile > 1:
            code.append("}")
        code += ["}", "__syncthreads();"]
    if asynchronous:
        code.append("stage^=1;")
    code.append("}")
    if not mixed:
        slot = next(slot for _, _, pairs in terms for role, slot in pairs if role == 4)
        pointer = next(mapping[4] for mapping, _, pairs in terms if (4, slot) in pairs)
        code.append("switch(warp){")
        for i, path in enumerate(paths):
            code.append(f"case {i}:{{")
            for m in range(path[7]):
                code.append(
                    f"g{slot}[node*{dims[pointer]}+{path[2] + m * 128}+u]=sum{m};"
                )
            code.append("break;}")
        code.append("}")
    code.append("}")
    return "\n".join(code)


@lru_cache(maxsize=256)
def cooperative_plan(
    metadata, program, dims, shared, dtype, device, tile, asynchronous, row_tile=1
):
    paths = parse_metadata(metadata)[0]
    groups = defaultdict(list)
    for path in paths:
        groups[path[0]].append(path)
    sources = []
    for group in groups.values():
        group.sort(key=lambda p: len(p[-1]))
        for begin in range(0, len(group), tile):
            selected = tuple(group[begin : begin + tile])
            code = cooperative_source(
                selected, program, dims, shared, dtype, asynchronous, row_tile
            )
            packing = packed_description(metadata)
            if packing is not None:
                code = pack_edge_source(code, packing, program)
            sources.append((code, len(selected) * 32))
    compiled = kernels([code for code, _ in sources], device)
    return tuple((compiled[code], threads) for code, threads in sources)


def cooperative_edges(
    metadata, program, source, target, operands, tile=4, asynchronous=False, row_tile=1
):
    """Evaluate all production paths with shared mutable graph/model operands."""
    terms = parse_metadata(program)
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    if (
        roles not in ({4}, {0, 1})
        or parse_metadata(metadata)[2]
        or tile not in (1, 2, 4, 8)
    ):
        raise ValueError("Unsupported cooperative phase or tile")
    if any(operands[m[1]].shape[0] == 1 or operands[m[2]].numel() for m, _, _ in terms):
        raise ValueError("Cooperative paths require independent edge weights")
    outputs = o3_outputs(metadata, program, source, target, operands)
    zero = {
        slot
        for _, _, pairs in terms
        for role, slot in pairs
        if role == 0 or source.numel() == 0
    }
    for slot in zero:
        outputs[slot].zero_()
    if source.numel() == 0:
        return outputs
    required = {
        i
        for m, _, pairs in terms
        for output, _ in pairs
        for role, i in enumerate(m)
        if role != output
    }
    operands = [v.contiguous() if i in required else v for i, v in enumerate(operands)]
    source, target = source.contiguous(), target.contiguous()
    nodes = operands[terms[0][0][4]].shape[0]
    order, pointers = prepare_graph(source, target, 1, nodes)
    plan = cooperative_plan(
        metadata,
        terms,
        tuple(v.shape[1] for v in operands),
        tuple(v.shape[0] == 1 for v in operands),
        "float" if operands[0].dtype == torch.float32 else "double",
        operands[0].device,
        tile,
        asynchronous,
        row_tile,
    )
    arguments = [v.data_ptr() for v in (*operands, *outputs)]
    arguments += [
        source.data_ptr(),
        target.data_ptr(),
        order.data_ptr(),
        pointers.data_ptr(),
        nodes,
    ]
    if packed_description(metadata) is not None:
        arguments.append(nodes)
    runtime().launch(
        [(kernel, arguments, nodes, 4, threads, 0) for kernel, threads in plan],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs
