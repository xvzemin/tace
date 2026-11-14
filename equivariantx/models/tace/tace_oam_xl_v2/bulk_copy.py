"""Stage contiguous channel spans using Hopper bulk asynchronous copies."""

from collections import defaultdict
from functools import lru_cache
import re


@lru_cache(maxsize=128)
def bulk_stage_source(source, dtype):
    """Preserve CG expressions while double buffering complete neighbor rows.

    Bulk transfers own channel-aligned spans. Scalar spherical-harmonic loads
    use ordinary shared stores. Two mbarriers track only bulk-transfer bytes;
    CTA synchronization also makes the scalar stores visible before use.
    """
    start = source.index("for (int64_t position =")
    opening = source.index("{", start)
    end = opening + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    body = source[opening + 1 : end - 1].replace(
        "const int64_t edge = order[position];", ""
    )
    expressions = list(dict.fromkeys(re.findall(r"p\d+\[[^\]]+\]", body)))
    groups = defaultdict(dict)
    scalars = []
    pattern = (
        r"p(\d+)\[\(?(source_node|target_node|edge)\)?\*(\d+)\+(\d+)\+u(?:\*1\+0)?\]"
    )
    for expression in expressions:
        if not re.search(r"\bu\b", expression):
            scalars.append(expression)
            continue
        match = re.fullmatch(pattern, re.sub(r"\s+", "", expression))
        if not match:
            raise ValueError("Unsupported bulk channel address: " + expression)
        slot, row, stride, column = match.groups()
        column = int(column)
        if column % 128:
            raise ValueError("Bulk channels must be aligned to 128 elements")
        groups[int(slot), row, int(stride)][column] = expression
    spans = []
    mapping = {}
    elements = 0
    for (slot, row, stride), columns in sorted(groups.items()):
        ordered = sorted(columns)
        first = 0
        while first < len(ordered):
            stop = first + 1
            while stop < len(ordered) and ordered[stop] == ordered[stop - 1] + 128:
                stop += 1
            column = ordered[first]
            count = ordered[stop - 1] + 128 - column
            spans.append((slot, row, stride, column, elements, count))
            for c in ordered[first:stop]:
                mapping[columns[c]] = f"staging[stage][{elements + c - column}+u]"
            elements += count
            first = stop
    bulk_elements = elements
    for expression in scalars:
        mapping[expression] = f"staging[stage][{elements}]"
        elements += 1
    elements = (elements + 127) // 128 * 128
    itemsize = 4 if dtype == "float" else 8
    if elements * 2 * itemsize + 16 > 46000:
        return None
    if not spans:
        raise ValueError("No contiguous channel spans to transfer")
    for expression, replacement in mapping.items():
        body = body.replace(expression, replacement)

    def preload(edge, stage):
        lines = [
            "if(threadIdx.x==0) {",
            f"const int64_t edge={edge};",
            "const int64_t source_node=source[edge],target_node=target[edge];",
            f"const unsigned int barrier=static_cast<unsigned int>(__cvta_generic_to_shared(&barriers[{stage}]));",
            f'asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(barrier), "r"({bulk_elements * itemsize}) : "memory");',
        ]
        for slot, row, stride, column, offset, count in spans:
            lines += [
                "{",
                f"const unsigned int destination=static_cast<unsigned int>(__cvta_generic_to_shared(&staging[{stage}][{offset}]));",
                f"const T* origin=p{slot}+{row}*{stride}+{column};",
                'asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"',
                f':: "r"(destination), "l"(origin), "r"({count * itemsize}), "r"(barrier) : "memory");',
                "}",
            ]
        for i, expression in enumerate(scalars):
            lines += [f"staging[{stage}][{bulk_elements + i}]={expression};"]
        return "\n".join((*lines, "}"))

    loop = [
        f"__shared__ __align__(16) T staging[2][{elements}];",
        "__shared__ __align__(8) unsigned long long barriers[2];",
        "if(threadIdx.x==0) {",
        "for(int i=0;i<2;++i) {",
        "unsigned int address=static_cast<unsigned int>(__cvta_generic_to_shared(&barriers[i]));",
        'asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");',
        "}",
        'asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
        "}",
        "__syncthreads();",
        "int stage=0;",
        "if(begin<end) {",
        preload("order[begin]", "0"),
        "}",
        "for(int64_t position=begin;position<end;++position) {",
        "const unsigned int barrier=static_cast<unsigned int>(__cvta_generic_to_shared(&barriers[stage]));",
        "const unsigned int phase=static_cast<unsigned int>((position-begin)/2)&1;",
        "unsigned int ready=0;",
        'do { asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"',
        ': "=r"(ready) : "r"(barrier), "r"(phase) : "memory"); } while(!ready);',
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
