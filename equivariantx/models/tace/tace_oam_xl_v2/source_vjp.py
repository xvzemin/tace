"""Joint source, radial and angular edge adjoints in one CSR traversal.

The source feature is resident while its neighbors are traversed. Native CG
expressions share cotangent work between all three outputs. This alternative
trades receiver-cotangent locality for fewer repeated loads across adjoints.
"""

from .precision import require_unscaled, require_tensors
from collections import defaultdict, deque
from functools import lru_cache
import os
import re

import torch

from .cg_codegen import convolution_source
from .graph import prepare_graph
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .packed_edge import pack_edge_source, packed_description


@lru_cache(maxsize=128)
def block_owned_source(source):
    """Use a 128-thread source owner and one angular store per edge/group."""
    source = source.replace(
        "const int64_t* order,", "const int64_t* order, const int64_t* pointers,"
    )
    source = source.replace(
        "const int u = int(blockIdx.y) * 32 + lane;", "const int u = threadIdx.x;"
    )
    source = source.replace(
        "const int64_t task = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;",
        "const int64_t task = blockIdx.x;",
    )
    begin = source.index("const int64_t stop =")
    end = source.index("\n", source.index("const bool exclusive =", begin)) + 1
    source = (
        source[:begin]
        + "const int64_t node=task;\n"
        + "const int64_t begin=pointers[node], end=pointers[node+1];\n"
        + "const bool exclusive=true;\n"
        + source[end:]
    )
    if not source.endswith("\n}\n}"):
        raise ValueError("Unexpected segmented CSR source ending")
    source = source[:-2]
    pattern = r"if \(lane == 0\) atomicAdd\((g\d+ \+ edge \* \d+ \+ \d+), value\); \}"
    targets = re.findall(pattern, source)
    if not targets:
        return source
    source = source.replace(
        "if (task >= tasks) return;",
        f"if (task >= tasks) return;\n__shared__ T angular_sums[4][{len(targets)}];",
    )
    counter = iter(range(len(targets)))

    def replace(match):
        index = next(counter)
        text = f"if (lane == 0) angular_sums[threadIdx.x/32][{index}]=value; }}"
        if index == len(targets) - 1:
            text += "\n__syncthreads();\n"
            for i, address in enumerate(targets):
                total = "+".join(f"angular_sums[{warp}][{i}]" for warp in range(4))
                text += f"if(threadIdx.x=={i}) *({address}) += {total};\n"
            text += "__syncthreads();\n"
        return text

    return re.sub(pattern, replace, source)


@lru_cache(maxsize=128)
def staged_source(source, dtype):
    """Prefetch one neighbor while the block evaluates the current CG terms."""
    start = source.index("for (int64_t position =")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    body = source[opening + 1 : end - 1]
    body = body.replace("const int64_t edge = order[position];", "")
    expressions = list(dict.fromkeys(re.findall(r"p\d+\[[^\]]+\]", body)))
    slots = {}
    elements = 0
    for expression in expressions:
        width = 128 if re.search(r"\bu\b", expression) else 1
        slots[expression] = (elements, width)
        elements += width
    elements = (elements + 127) // 128 * 128
    if elements * 2 * (4 if dtype == "float" else 8) > 46000:
        return None
    for expression, (offset, width) in slots.items():
        body = body.replace(
            expression, f"staging[stage][{offset}" + ("+u]" if width == 128 else "]")
        )

    def preload(edge, stage):
        lines = [
            "{",
            f"const int64_t edge={edge};",
            "const int64_t source_node=source[edge],target_node=target[edge];",
        ]
        for expression, (offset, width) in slots.items():
            address = str(offset) + ("+u" if width == 128 else "")
            lines.extend(
                (
                    f"if(u<{width}) {{ unsigned int address=static_cast<unsigned int>(__cvta_generic_to_shared(&staging[{stage}][{address}]));",
                    f'asm volatile("cp.async.ca.shared.global [%0], [%1], {4 if dtype == "float" else 8};" :: "r"(address), "l"(&{expression})); }}',
                )
            )
        return "\n".join((*lines, 'asm volatile("cp.async.commit_group;");', "}"))

    loop = [
        f"__shared__ T staging[2][{elements}];",
        "int stage=0;",
        "if(begin<end) {",
        preload("order[begin]", "0"),
        "}",
        "for(int64_t position=begin;position<end;++position) {",
        'asm volatile("cp.async.wait_group 0;");',
        "__syncthreads();",
        "if(position+1<end) {",
        preload("order[position+1]", "stage^1"),
        "}",
        "const int64_t edge=order[position];",
        body,
        "__syncthreads();",
        "stage^=1;",
        "}",
    ]
    return source[:start] + "\n".join(loop) + source[end:]


@lru_cache(maxsize=128)
def source_plan(
    metadata,
    program,
    dimensions,
    shared,
    dtype,
    device,
    grouping,
    full_row,
    radial_scale,
    owner,
    asynchronous,
    register_limit=255,
    reuse_angular=None,
    vector_width=1,
    row_limit=0,
):
    require_unscaled(radial_scale, "radial_scale")
    paths = parse_metadata(metadata)[0]
    terms = parse_metadata(program)
    outputs = 1 + max(slot for _, _, pairs in terms for _, slot in pairs)
    weight_slots = {slot for _, _, pairs in terms for role, slot in pairs if role == 1}
    groups = defaultdict(list)
    for path in paths:
        groups[path[0] if grouping == "input" else path[1]].append(path)
    pending = deque(tuple(group) for group in groups.values())
    result = []
    while pending:
        group = pending.popleft()
        source = convolution_source(
            group,
            terms,
            dimensions,
            shared,
            dtype,
            owner,
            outputs,
            initialize=weight_slots,
            hoist_owner=True,
            reuse_angular=reuse_angular,
        )
        if full_row:
            source = block_owned_source(source)
        if asynchronous:
            if asynchronous == "bulk":
                from .bulk_copy import bulk_stage_source

                source = bulk_stage_source(source, dtype)
            else:
                source = staged_source(source, dtype)
            if source is None:
                if len(group) == 1:
                    raise ValueError("Single path exceeds the staging limit")
                split = len(group) // 2
                pending.extend((group[:split], group[split:]))
                continue
        packing = packed_description(metadata)
        if packing is not None:
            source = pack_edge_source(source, packing, terms)
        if vector_width != 1:
            if radial_scale or not full_row:
                raise ValueError("Vector channels require an unscaled complete row")
            from .channel_vectors import vectorize_channels

            source = vectorize_channels(source, terms, vector_width)
        if row_limit:
            if not full_row or asynchronous:
                raise ValueError(
                    "Segmented source adjoints require synchronous complete CTAs"
                )
            from .row_schedule import split_source

            source = split_source(source, row_limit)
        kernel = kernels([source], device)[source]
        if (kernel.local_bytes or kernel.registers > register_limit) and len(group) > 1:
            split = len(group) // 2
            pending.extend((group[:split], group[split:]))
            continue
        threads = (
            128 // vector_width
            if full_row
            else max(
                (64, 128, 256),
                key=lambda n: (n * kernel.active_blocks(n, 0), -abs(n - 128)),
            )
        )
        result.append((kernel, group[0][3], threads))
    return tuple(result)


def source_vjp_edges(
    metadata,
    program,
    source,
    target,
    operands,
    row_size=16,
    grouping="input",
    full_row=False,
    radial_scale=0,
    owner=0,
    asynchronous=False,
    vector_width=1,
):
    """Evaluate complete non-shared edge adjoints without changing terms."""
    require_unscaled(radial_scale, "radial_scale")
    require_tensors(operands)
    paths, _, weighted, *_ = parse_metadata(metadata)
    terms = parse_metadata(program)
    if asynchronous and not full_row:
        raise ValueError("Asynchronous staging requires a complete block-owned row")
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    row_limit = (
        int(
            os.environ.get(
                "EQX_XL_MIXED_ROW_LIMIT"
                if roles == {0, 1}
                else "EQX_XL_SOURCE_ROW_LIMIT",
                "0",
            )
        )
        if full_row
        else 0
    )
    if (
        roles not in ({0, 1, 3}, {0, 1})
        or owner not in (0, 1)
        or weighted
        or any(p[3] != 128 or p[4] != 1 or p[8] < 0 for p in paths)
    ):
        raise ValueError("Expected complete UVU adjoints with 128 channels")
    if any(operands[m[1]].shape[0] == 1 or operands[m[2]].numel() for m, _, _ in terms):
        raise ValueError("Source VJP requires explicit non-shared radial inputs")
    outputs = o3_outputs(metadata, program, source, target, operands)
    # All source and angular reductions require initialization. Weight columns
    # are assigned once by disjoint complete paths in the production contract.
    slots = {slot for _, _, pairs in terms for role, slot in pairs if role == 1}
    offsets = sorted((p[8], p[3]) for p in paths)
    if offsets[0][0] != 0 or any(
        a + c != b for (a, c), (b, _) in zip(offsets, offsets[1:])
    ):
        raise ValueError("Radial columns must form a complete nonoverlapping partition")
    for slot, out in enumerate(outputs):
        if slot not in slots or not source.numel():
            out.zero_()
        elif offsets[-1][0] + offsets[-1][1] != out.shape[1]:
            raise ValueError("Radial partition does not cover the result")
    if not source.numel():
        return outputs
    required = {
        i
        for m, _, pairs in terms
        for output, _ in pairs
        for role, i in enumerate(m)
        if role != output
    }
    values = [
        value.contiguous() if i in required else value
        for i, value in enumerate(operands)
    ]
    source, target = source.contiguous(), target.contiguous()
    nodes = values[terms[0][0][0 if owner == 0 else 4]].shape[0]
    order, pointers = prepare_graph(source, target, owner, nodes)
    dtype = "float" if values[0].dtype == torch.float32 else "double"
    plan = source_plan(
        metadata,
        program,
        tuple(x.shape[1] for x in values),
        tuple(x.shape[0] == 1 for x in values),
        dtype,
        values[0].device,
        grouping,
        full_row,
        radial_scale,
        owner,
        asynchronous,
        int(os.environ.get("EQX_XL_SOURCE_VJP_REGISTER_LIMIT", "255")),
        True
        if roles == {0, 1} and os.environ.get("EQX_XL_MIXED_REUSE_ANGULAR", "0") == "1"
        else None,
        vector_width,
        row_limit,
    )
    tasks = nodes if full_row else (source.numel() + row_size - 1) // row_size
    args = [x.data_ptr() for x in (*values, *outputs)]
    args += [source.data_ptr(), target.data_ptr(), order.data_ptr()]
    if full_row:
        args.append(pointers.data_ptr())
    args += [source.numel(), tasks, row_size]
    if packed_description(metadata) is not None:
        args.append(nodes)
    if row_limit:
        from .row_schedule import prepare_segments

        jobs = prepare_segments(pointers, row_limit)
        args.append(jobs.data_ptr())
        tasks = nodes + (source.numel() + row_limit - 1) // row_limit
    runtime().launch(
        [
            (
                kernel,
                args,
                tasks if full_row else (tasks + threads // 32 - 1) // (threads // 32),
                1 if full_row else (width + 31) // 32,
                threads,
                0,
            )
            for kernel, width, threads in plan
        ],
        torch.cuda.current_stream(values[0].device).cuda_stream,
    )
    return outputs
