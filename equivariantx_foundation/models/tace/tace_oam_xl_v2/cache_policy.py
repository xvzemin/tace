"""Optional PTX cache priorities for streamed radial and reused node operands.

Cache hints affect residency only; arithmetic, indexing and memory ordering
remain unchanged. SM80+ fractional L2 hints are documented in NVIDIA PTX ISA.
"""

import os
import re

POLICY = os.environ.get("EQX_XL_CACHE_POLICY", "normal")


def source_policy(source, program, dtype):
    """Apply checked read-only input hints to one native source string."""
    if POLICY == "normal":
        return source
    if POLICY not in ("radial_cg", "node_last", "stream_first", "split_priority"):
        raise ValueError(f"Unknown cache policy {POLICY}")
    roles = {}
    for mapping, _, pairs in program:
        for role, pointer in enumerate(mapping):
            roles.setdefault(pointer, set()).add(role)
    names = {}
    helpers = []
    suffix, constraint = ("f32", "f") if dtype == "float" else ("f64", "d")
    for pointer, kinds in roles.items():
        style = None
        if kinds == {1} and POLICY in ("radial_cg", "stream_first", "split_priority"):
            style = "cg" if POLICY == "radial_cg" else "first"
        elif kinds <= {0, 4} and POLICY in ("node_last", "split_priority"):
            style = "last"
        if style is None:
            continue
        name = "xl_load_" + style
        names[pointer] = name
        if any(name in h for h in helpers):
            continue
        policy = ""
        instruction = "ld.global.cg." + suffix
        if style != "cg":
            policy = f'unsigned long long policy; asm("createpolicy.fractional.L2::evict_{style}.b64 %0, 1.0;" : "=l"(policy));'
            instruction = f"ld.global.L1::evict_{style}.L2::cache_hint.{suffix}"
        hint = ", %2" if style != "cg" else ""
        operand = ', "l"(policy)' if style != "cg" else ""
        helpers.append(
            f'__device__ __forceinline__ T {name}(const T* p) {{ T value; {policy} asm("{instruction} %0, [%1]{hint};" : "={constraint}"(value) : "l"(p){operand}); return value; }}'
        )
    source = re.sub(
        r"p(\d+)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]",
        lambda m: (
            f"{names[int(m[1])]}(p{m[1]}+({m[2]}))" if int(m[1]) in names else m[0]
        ),
        source,
    )
    marker = f"using T = {dtype};"
    if marker not in source:
        raise ValueError("Missing native scalar declaration")
    return source.replace(marker, marker + "\n" + "\n".join(helpers), 1)
