"""Receiver-owned forward and tangent CG with optional asynchronous staging.

See E3J Appendix E.3, https://arxiv.org/html/2609.35099v1 . This implementation
keeps every independent path weight and recursively differentiated term.
"""

from collections import defaultdict, deque
from functools import lru_cache
import re
import torch

from .cg_codegen import convolution_source
from .graph import prepare_graph
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .packed_edge import packed_description, pack_edge_source


def generate(paths, terms, dims, shared, dtype, asynchronous, owner=4):
    """Keep generated CG expressions and give each receiver one 128-thread CTA."""
    outputs = 1 + max(slot for _, _, pairs in terms for _, slot in pairs)
    assert owner in (0, 4) and all(
        role == owner for _, _, pairs in terms for role, _ in pairs
    )
    assert paths[0][3] == 128
    source = convolution_source(
        paths, terms, dims, shared, dtype, int(owner == 4), outputs
    )
    beginning = source.index("const int64_t stop =")
    loop = source.index("for (int64_t position =")
    first = source.index("const int64_t source_node", loop)
    final = source.index("\n}\nif (active)", first)
    sums = "\n".join(re.findall(r"T sum\d+ = 0;", source[beginning:loop]))
    body = source[first:final]
    stores = source[final + 3 : source.rfind("\n}\n}")]
    stores = re.sub(
        r"if \(active\) \{ if \(exclusive\) (g\d+\[[^\]]+\]) \+= (sum\d+); else atomicAdd\(&[^;]+; \}",
        r"if (active) \1 = \2;",
        stores,
    )
    if owner == 0:
        stores = re.sub(r"(g\d+\[[^\]]+\]) = (sum\d+);", r"\1 += \2;", stores)
    if "atomicAdd" in stores or "exclusive" in stores:
        raise ValueError("Unexpected reduction store contract")
    header = source[:beginning]
    header = header.replace(
        "const int64_t* order,", "const int64_t* order, const int64_t* pointers,"
    )
    header = header.replace(
        "const int u = int(blockIdx.y) * 32 + lane;", "const int u = threadIdx.x;"
    )
    header = header[: header.index("const int64_t task =")]
    code = [
        header,
        "const int64_t node = blockIdx.x;",
        "if (node >= tasks) return;",
        "const int64_t begin=pointers[node], end=pointers[node+1];",
        sums,
    ]
    if asynchronous:
        expressions = list(dict.fromkeys(re.findall(r"p\d+\[[^\]]+\]", body)))
        slots = {}
        elements = 0
        for expression in expressions:
            width = 128 if re.search(r"\bu\b", expression) else 1
            slots[expression] = (elements, width)
            elements += width
        elements = (elements + 127) // 128 * 128
        for expression, (offset, width) in slots.items():
            body = body.replace(
                expression,
                f"staging[stage][{offset}" + ("+u]" if width == 128 else "]"),
            )
        code += [f"__shared__ T staging[2][{elements}];", "int stage=0;"]

        def preload(edge, stage):
            lines = [
                "{",
                f"const int64_t edge={edge};",
                "const int64_t source_node=source[edge], target_node=target[edge];",
            ]
            for expression, (offset, width) in slots.items():
                address = str(offset) + ("+u" if width == 128 else "")
                lines += [
                    f"if (u<{width}) {{ unsigned int address=static_cast<unsigned int>(__cvta_generic_to_shared(&staging[{stage}][{address}]));",
                    f'asm volatile("cp.async.ca.shared.global [%0], [%1], {4 if dtype == "float" else 8};" :: "r"(address), "l"(&{expression})); }}',
                ]
            lines += ['asm volatile("cp.async.commit_group;");', "}"]
            return "\n".join(lines)

        code += [
            "if (begin < end) {",
            preload("order[begin]", "0"),
            "}",
            "for (int64_t position=begin;position<end;++position) {",
            'asm volatile("cp.async.wait_group 0;");',
            "__syncthreads();",
            "if (position+1 < end) {",
            preload("order[position+1]", "stage^1"),
            "}",
            "const int64_t edge=order[position];",
            body,
            "__syncthreads();",
            "stage^=1;",
            "}",
        ]
    else:
        code += [
            "for (int64_t position=begin;position<end;++position) {",
            "const int64_t edge=order[position];",
            body,
            "}",
        ]
    code += [stores, "}"]
    return "\n".join(code)


@lru_cache(maxsize=256)
def plan(
    metadata,
    program,
    dims,
    shared,
    dtype,
    device,
    asynchronous,
    radial_indexed=False,
    owner=4,
    vector_width=1,
):
    paths = parse_metadata(metadata)[0]
    terms = parse_metadata(program)
    groups = defaultdict(list)
    for p in paths:
        groups[p[0]].append(p)
    pending = deque(tuple(p) for p in groups.values())
    result = []
    while pending:
        paths = pending.popleft()
        code = generate(paths, terms, dims, shared, dtype, asynchronous, owner)
        packing = packed_description(metadata)
        if packing is not None:
            code = pack_edge_source(code, packing, terms)
        if radial_indexed:
            from .compressed import compressed_source

            code = compressed_source(code, terms)
        if vector_width != 1:
            from .channel_vectors import vectorize_channels

            code = vectorize_channels(code, terms, vector_width)
        storage = re.search(r"T staging\[2\]\[(\d+)\]", code)
        if (
            storage
            and int(storage[1]) * 2 * (4 if dtype == "float" else 8) > 48000
            and len(paths) > 1
        ):
            half = len(paths) // 2
            pending.extend([paths[:half], paths[half:]])
            continue
        kernel = kernels([code], device)[code]
        if (kernel.local_bytes or kernel.registers > 255) and len(paths) > 1:
            half = len(paths) // 2
            pending.extend([paths[:half], paths[half:]])
            continue
        result.append((kernel, paths))
    return result


def receiver_edges(
    metadata,
    program,
    source,
    target,
    operands,
    asynchronous=True,
    radial_map=None,
    vector_width=1,
):
    """Evaluate a complete pure-receiver program, including mixed tangents.

    Parameters
    ----------
    metadata, program : str
        Native explicit UVU paths and already differentiated terms.
    source, target : torch.Tensor
        Live int64 edge indices. Empty and changing graphs are supported.
    operands : list of torch.Tensor
        Native input buffers, including placeholders for unused operands.
    asynchronous : bool
        Stage edge inputs in two shared-memory buffers when True.
    """
    meta, terms = parse_metadata(metadata), parse_metadata(program)
    if vector_width != 1 and (asynchronous or operands[0].dtype != torch.float32):
        raise ValueError("Vector channels require synchronous FP32 memory access")
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    if (
        roles not in ({0}, {4})
        or meta[2]
        or any(p[3] != 128 or p[4] != 1 or p[8] < 0 for p in meta[0])
    ):
        raise ValueError(
            "Receiver scheduling requires 128-channel explicit weighted UVU paths"
        )
    owner = next(iter(roles))
    results = o3_outputs(metadata, program, source, target, operands)
    if owner == 0:
        for result in results:
            result.zero_()
    if not source.numel() or not meta[0]:
        for result in results:
            result.zero_()
        return results
    required = {
        index
        for mapping, _, pairs in terms
        for output, _ in pairs
        for role, index in enumerate(mapping)
        if role != output
    }
    operands = [
        value.contiguous() if i in required else value
        for i, value in enumerate(operands)
    ]
    source, target = source.contiguous(), target.contiguous()
    nodes = operands[terms[0][0][owner]].shape[0]
    order, pointers = prepare_graph(source, target, int(owner == 4), nodes)
    stages = plan(
        metadata,
        program,
        tuple(x.shape[1] for x in operands),
        tuple(x.shape[0] == 1 for x in operands),
        "float" if operands[0].dtype == torch.float32 else "double",
        operands[0].device,
        asynchronous,
        radial_map is not None,
        owner,
        vector_width,
    )
    arguments = [v.data_ptr() for v in (*operands, *results)] + [
        source.data_ptr(),
        target.data_ptr(),
        order.data_ptr(),
        pointers.data_ptr(),
        source.numel(),
        nodes,
        1,
    ]
    if packed_description(metadata) is not None:
        arguments.append(nodes)
    if radial_map is not None:
        arguments.append(radial_map.data_ptr())
    runtime().launch(
        [(kernel, arguments, nodes, 1, 128 // vector_width, 0) for kernel, _ in stages],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return results
