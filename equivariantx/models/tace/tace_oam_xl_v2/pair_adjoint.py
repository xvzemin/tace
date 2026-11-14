"""Exact periodic-image factoring for source and radial mixed adjoints.

All images of an ordered atom pair have the same node features/cotangents.
Radial adjoints first contract those two node operands. Source adjoints first
sum the radial-times-angular factors over images. Every image still receives
its own radial gradient. This experimental schedule changes only association.
"""

from collections import defaultdict
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.codegen import HEADER, contraction_source
from .cg_codegen import angular_source
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .packed_edge import packed_description, pack_edge_source


def pair_order(source, target, nodes):
    """Build a sorted pair plan outside capture for isolated kernel probes."""
    key = target * nodes + source
    order = key.argsort(stable=True)
    ordered = key[order]
    starts = torch.cat(
        (ordered.new_ones(1, dtype=torch.bool), ordered[1:] != ordered[:-1])
    )
    ptr = torch.cat((starts.nonzero().flatten(), order.new_tensor([order.numel()])))
    return order, ptr


@lru_cache(maxsize=256)
def pair_source(paths, program, dimensions, shared, dtype, outputs):
    """Emit one warp per ordered atom pair and channel slice."""
    args = [f"const T* p{i}" for i in range(len(dimensions))]
    args += [f"T* g{i}" for i in range(outputs)]
    args += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* order",
        "const int64_t* ptr",
        "int64_t pairs",
    ]
    head = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ",".join(args) + ") {",
        "const int64_t pair=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;",
        "if(pair>=pairs)return;",
        "const int u=blockIdx.y*32+threadIdx.x%32;",
        "constexpr bool active=true;",
        "const int64_t begin=ptr[pair], end=ptr[pair+1];",
        "const int64_t first=order[begin];",
        "const int64_t source_node=source[first], target_node=target[first];",
    ]
    cache, before, inside, after = {}, [], [], []
    moments, node_terms, radial_terms = {}, defaultdict(list), defaultdict(list)

    def yload(pointer, column):
        key = ("edge_y", pointer, column)
        if key not in cache:
            name = f"v{len(cache)}"
            inside.append(
                f"const T {name}=p{pointer}[edge*{dimensions[pointer]}+{column}];"
            )
            cache[key] = name
        return cache[key]

    for path in paths:
        start, attr, stop, width, _, dimx, dimy, dimz, weight, factor, cg = path
        for mapping, weighted, results in program:
            for role, slot in results:
                if role == 1:
                    angular = angular_source(
                        path, mapping, dimensions, shared, {3}, cache, before
                    )
                    terms = [
                        (1.0, (angular[0, 3, b], yload(mapping[3], attr + b)))
                        for b in range(dimy)
                    ]
                    result = contraction_source(terms, cache, inside)
                    radial_terms[slot, weight].append((factor, (result,)))
                elif role == 0:
                    for b in range(dimy):
                        key = mapping[1], mapping[3], weight, attr + b
                        if key not in moments:
                            name = f"q{len(moments)}"
                            moments[key] = name
                            before.append(f"T {name}=0;")
                            wkey = "edge_w", mapping[1], weight
                            if wkey not in cache:
                                wname = f"v{len(cache)}"
                                inside.append(
                                    f"const T {wname}=p{mapping[1]}[edge*{dimensions[mapping[1]]}+{weight}+u];"
                                )
                                cache[wkey] = wname
                            inside.append(
                                f"{name}=fma({cache[wkey]},{yload(mapping[3], attr + b)},{name});"
                            )
                    for a, b, z, coefficient in cg:
                        key = mapping[4], "target_node", stop + z * width, "u"
                        if key not in cache:
                            name = f"v{len(cache)}"
                            before.append(
                                f"const T {name}=p{mapping[4]}[(target_node)*{dimensions[mapping[4]]}+{stop + z * width}+u];"
                            )
                            cache[key] = name
                        node_terms[slot, start + a * width].append(
                            (
                                factor * coefficient,
                                (
                                    cache[key],
                                    moments[mapping[1], mapping[3], weight, attr + b],
                                ),
                            )
                        )
                else:
                    raise ValueError(
                        "Pair adjoints currently require source/radial outputs"
                    )
    for (slot, weight), terms in radial_terms.items():
        value = contraction_source(terms, cache, inside)
        inside.append(
            f"g{slot}[edge*{dimensions[program[0][0][1]]}+{weight}+u]={value};"
        )
    for (slot, column), terms in node_terms.items():
        value = contraction_source(terms, cache, after)
        after.append(
            f"atomicAdd(g{slot}+source_node*{dimensions[program[0][0][0]]}+{column}+u,{value});"
        )
    return "\n".join(
        head
        + before
        + [
            "for(int64_t position=begin;position<end;++position){",
            "const int64_t edge=order[position];",
        ]
        + inside
        + ["}"]
        + after
        + ["}"]
    )


@lru_cache(maxsize=256)
def pair_plan(metadata, program, dimensions, shared, dtype, device, tile):
    """Compile bounded path tiles; callers can inspect resources and tune."""
    paths = parse_metadata(metadata)[0]
    groups = defaultdict(list)
    for path in paths:
        groups[path[0]].append(path)
    count = 1 + max(slot for _, _, results in program for _, slot in results)
    source_codes = []
    for group in groups.values():
        for start in range(0, len(group), tile):
            code = pair_source(
                tuple(group[start : start + tile]),
                program,
                dimensions,
                shared,
                dtype,
                count,
            )
            packing = packed_description(metadata)
            if packing is not None:
                code = pack_edge_source(code, packing, program)
            source_codes.append(code)
    compiled = kernels(source_codes, device)
    return tuple(compiled[code] for code in source_codes)


def pair_adjoints(metadata, program, source, target, operands, order, ptr, tile=2):
    """Compute complete mixed adjoints using a caller-owned live pair plan."""
    paths, _, weighted, *_ = parse_metadata(metadata)
    terms = parse_metadata(program)
    if weighted or any(p[3] != 128 or p[4] != 1 or p[8] < 0 for p in paths):
        raise ValueError("Pair adjoints require nonshared 128-channel UVU paths")
    roles = {role for _, _, results in terms for role, _ in results}
    if roles != {0, 1}:
        raise ValueError("Pair adjoints require complete source/radial outputs")
    if any(operands[m[1]].shape[0] == 1 or operands[m[2]].numel() for m, _, _ in terms):
        raise ValueError("Shared or auxiliary weights are unsupported")
    outputs = o3_outputs(metadata, program, source, target, operands)
    for _, _, results in terms:
        for role, slot in results:
            if role == 0:
                outputs[slot].zero_()
    if not source.numel():
        return [out.zero_() for out in outputs]
    values = [v.contiguous() for v in operands]
    dtype = "double" if values[0].dtype == torch.float64 else "float"
    plan = pair_plan(
        metadata,
        terms,
        tuple(v.shape[1] for v in values),
        tuple(v.shape[0] == 1 for v in values),
        dtype,
        values[0].device,
        tile,
    )
    count = ptr.numel() - 1
    args = [v.data_ptr() for v in (*values, *outputs, source, target, order, ptr)] + [
        count
    ]
    if packed_description(metadata) is not None:
        args.append(values[terms[0][0][4]].shape[0])
    runtime().launch(
        [(kernel, args, (count + 3) // 4, 4, 128, 0) for kernel in plan],
        torch.cuda.current_stream(values[0].device).cuda_stream,
    )
    return outputs
