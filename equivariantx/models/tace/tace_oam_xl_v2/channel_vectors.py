"""Pack adjacent FP32 channels into native aligned loads and stores.

The arithmetic still uses one FP32 FMA per component. Complete-row programs
reduce scalar angular outputs across both lanes and vector components.
"""

from functools import lru_cache
import re


@lru_cache(maxsize=16)
def vector_header(width):
    """Generate a trivial aligned value type with componentwise FP32 math."""
    if width not in (2, 4):
        raise ValueError("Expected two or four channels per thread")
    names = "xyzw"[:width]
    lines = [
        f"struct alignas({4 * width}) T {{",
        "float " + ",".join(names) + ";",
        "__device__ __forceinline__ T() = default;",
        "__device__ __forceinline__ T(float v):"
        + ",".join(f"{c}(v)" for c in names)
        + " {}",
        "__device__ __forceinline__ T& operator+=(T b) {",
    ]
    lines += [f"{c} += b.{c};" for c in names]
    lines += ["return *this; }", "};"]
    for op in ("+", "-", "*"):
        lines += [f"__device__ __forceinline__ T operator{op}(T a,T b) {{ T r;"]
        lines += [f"r.{c}=a.{c}{op}b.{c};" for c in names]
        lines += ["return r; }"]
    lines += ["__device__ __forceinline__ T operator-(T a) { T r;"]
    lines += [f"r.{c}=-a.{c};" for c in names]
    lines += ["return r; }", "__device__ __forceinline__ T fma(T a,T b,T c) { T r;"]
    lines += [f"r.{c}=fmaf(a.{c},b.{c},c.{c});" for c in names]
    lines += ["return r; }", "__device__ __forceinline__ T warp_sum(T a) {"]
    lines += [
        f"for(int o=16;o;o>>=1) a.{c}+=__shfl_down_sync(0xffffffff,a.{c},o);"
        for c in names
    ]
    lines += [
        "return a; }",
        "__device__ __forceinline__ float channel_sum(T a) { return "
        + "+".join(f"a.{c}" for c in names)
        + "; }",
        "__device__ __forceinline__ T load_channels(const float* p) { return *reinterpret_cast<const T*>(p); }",
        "__device__ __forceinline__ void store_channels(float* p,T a) { *reinterpret_cast<T*>(p)=a; }",
        "__device__ __forceinline__ void add_channels(float* p,T a) { store_channels(p,load_channels(p)+a); }",
        "__device__ __forceinline__ void atomicAdd(float* p,T a) {",
    ]
    lines += [f"atomicAdd(p+{i},a.{c});" for i, c in enumerate(names)]
    lines += ["}"]
    return "\n".join(lines) + "\n"


def vectorize_channels(source, program, width):
    """Rewrite an aligned complete-row CG source with FP32 vector memory IO."""
    if width not in (2, 4) or "using T = float;" not in source:
        raise ValueError("Channel vectorization requires FP32 and width two or four")
    if "staging[" in source:
        raise ValueError("Asynchronous staging is unsupported")
    if any(role not in (0, 1, 3, 4) for _, _, pairs in program for role, _ in pairs):
        raise ValueError("Expected channel-valued outputs")
    if "const int u = threadIdx.x;" not in source:
        raise ValueError("Expected one complete row per block")
    roles = {}
    for mapping, _, _ in program:
        for role, slot in enumerate(mapping):
            roles.setdefault(slot, set()).add(role)
    for slot, values in roles.items():
        if 3 in values and values != {3}:
            raise ValueError("Scalar and vector inputs cannot share an operand")
    beginning = source.index('extern "C" __global__')
    source = "using int64_t = long long;\n" + vector_header(width) + source[beginning:]
    source = re.sub(r"\bT\* ([pg]\d+)\b", r"float* \1", source)
    source = source.replace(
        "const int u = threadIdx.x;", f"const int u = {width}*threadIdx.x;"
    )
    if "angular_sums[" in source:
        source = source.replace("T angular_sums[4]", f"T angular_sums[{4 // width}]")

        def angular_store(match):
            index = match[2]
            total = "+".join(f"angular_sums[{i}][{index}]" for i in range(4 // width))
            return f"*({match[1]}) += channel_sum({total});"

        source, count = re.subn(
            r"\*\((g\d+[^;]+)\) \+= angular_sums\[0\]\[(\d+)\][^;]+;",
            angular_store,
            source,
        )
        if not count:
            raise ValueError("Expected block-owned scalar angular stores")

    def read(match):
        slot, address = int(match[1]), match[2]
        if roles.get(slot) == {3}:
            return f"T(p{slot}[{address}])"
        if not re.search(r"\bu\b", address):
            raise ValueError("Expected a channel-aligned input address")
        return f"load_channels(p{slot}+({address}))"

    source = re.sub(r"\bp(\d+)\[([^\]]+)\]", read, source)

    def write(match):
        slot, address, op, value = match.groups()
        fn = "add_channels" if op == "+=" else "store_channels"
        return f"{fn}(g{slot}+({address}),{value});"

    source = re.sub(r"\bg(\d+)\[([^\]]+)\]\s*(\+=|=)\s*([^;]+);", write, source)
    source = re.sub(r"&g(\d+)\[([^\]]+)\]", r"(g\1+(\2))", source)
    if re.search(r"\bg\d+\[", source):
        raise ValueError("Unconverted vector output address")
    return source
