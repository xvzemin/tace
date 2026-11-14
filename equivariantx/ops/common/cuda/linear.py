"""CUDA indexed channel contractions and their transposes."""

from collections import defaultdict
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils.metadata import parse_metadata
from ....utils.outputs import linear_output
from ...precision import validate_uniform
from ..linear import backward, setup_context
from .runtime import kernels, runtime


@lru_cache(maxsize=128)
def weight_source(metadata, dtype):
    """Reuse feature tiles across entries of each coefficient gradient."""
    din, dout, numel, experts, paths, layout = metadata
    max_dim = min(8, max(path[5] for path in paths))
    rows = 16
    itemsize = 8 if dtype == torch.float64 else 4
    while rows > 1 and rows * max_dim * 40 * itemsize > 24 * 1024:
        rows //= 2
    cases, widths = [], []
    for group, (sx, sw, sy, ci, co, dim, scale) in enumerate(paths):
        angular = min(dim, max_dim)
        cols = (co + 31) // 32
        tiles = ((ci + 7) // 8) * cols
        widths.append(experts * tiles)
        xin = f"(e*{ci}+u)*{dim}+m" if layout == "mul_ir" else f"(m*{experts}+e)*{ci}+u"
        yout = (
            f"(e*{co}+v)*{dim}+m" if layout == "mul_ir" else f"(m*{experts}+e)*{co}+v"
        )
        cases.append(f"""
        case {group}: {{
            long long start=(long long)blockIdx.x/{experts * tiles}*{rows};
            if(start>=nodes) return;
            int e=(blockIdx.x/{tiles})%{experts};
            int tile=blockIdx.x%{tiles}, first_u=(tile/{cols})*8, first_v=(tile%{cols})*32;
            int tid=threadIdx.x;
            if(tid<{rows}) elements[tid]=start+tid<nodes ? types[order[start+tid]] : -1;
            for(int first_m=0;first_m<{dim};first_m+={angular}) {{
            for(int k=tid;k<{rows * angular * 8};k+=256) {{
                int r=k/{angular * 8},m=first_m+(k/8)%{angular},u=first_u+k%8;
                x[k]=(start+r<nodes && u<{ci} && m<{dim}) ? p0[order[start+r]*{din}+{sx}+{xin}] : scalar(0);
            }}
            for(int k=tid;k<{rows * angular * 32};k+=256) {{
                int r=k/{angular * 32},m=first_m+(k/32)%{angular},v=first_v+k%32;
                y[k]=(start+r<nodes && v<{co} && m<{dim}) ? p2[order[start+r]*{dout}+{sy}+{yout}] : scalar(0);
            }}
            __syncthreads();
            int u=first_u+tid/32,v=first_v+tid%32;
            if(u<{ci} && v<{co}) {{
            scalar total=0; long long previous=elements[0];
            for(int r=0;r<{rows} && start+r<nodes;++r) {{
                long long z=elements[r];
                if(z!=previous) {{
                    atomicAdd(result+(previous*{experts}+e)*{numel}+{sw}+u*{co}+v,scalar({scale:.17g})*total);
                    total=0; previous=z;
                }}
                #pragma unroll
                for(int m=0;m<{angular};++m) total += x[(r*{angular}+m)*8+tid/32]*y[(r*{angular}+m)*32+tid%32];
            }}
            atomicAdd(result+(previous*{experts}+e)*{numel}+{sw}+u*{co}+v,scalar({scale:.17g})*total);
            }}
            __syncthreads();
            }}
            return;
        }}
        """)
    scalar = "double" if dtype == torch.float64 else "float"
    code = f"""
    using scalar={scalar};
    extern "C" __global__ void run(const scalar* __restrict__ p0,
        const scalar* __restrict__ p1, const scalar* __restrict__ p2,
        const long long* __restrict__ types, const long long* __restrict__ order,
        scalar* __restrict__ result, long long nodes) {{
        __shared__ scalar x[{rows * max_dim * 8}],y[{rows * max_dim * 32}];
        __shared__ long long elements[{rows}];
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}
    """
    return code, max(widths), len(paths), rows


@lru_cache(maxsize=256)
def source(metadata, role, dtype, transposed):
    """Coalesce coefficient reads for forward and transposed channel maps."""
    din, dout, numel, experts, paths, layout = metadata
    groups = defaultdict(list)
    for path in paths:
        sx, sw, sy, ci, co, dim, scale = path
        key = (sx, ci * dim) if role == 0 else (sy, co * dim)
        groups[key].append(path)
    cases, widths = [], []
    for group, ((offset, width), entries) in enumerate(groups.items()):
        widths.append(experts * width)
        lines = [
            f"case {group}: {{",
            f"long long n=index/{experts * width}; if(n>=nodes) return;",
            f"int e=(index/{width})%{experts}, c=index%{width};",
            "long long z=types[n]; scalar total=0;",
        ]
        for sx, sw, sy, ci, co, dim, scale in entries:
            xin = (
                f"(e*{ci}+u)*{dim}+m"
                if layout == "mul_ir"
                else f"(m*{experts}+e)*{ci}+u"
            )
            yout = (
                f"(e*{co}+v)*{dim}+m"
                if layout == "mul_ir"
                else f"(m*{experts}+e)*{co}+v"
            )
            channels = ci if role == 0 else co
            lines += [
                "{",
                f"int {'u' if role == 0 else 'v'}=c%{channels},m=c/{channels}; scalar value=0;",
                "#pragma unroll 4",
                (
                    (
                        f"for(int v=0;v<{co};++v) {{"
                        if transposed
                        else f"for(int v=lane;v<{co};v+=32) {{"
                    )
                    if role == 0
                    else f"for(int u=0;u<{ci};++u) {{"
                ),
                (
                    f"value += p2[n*{dout}+{sy}+{yout}]"
                    if role == 0
                    else f"value += p0[n*{din}+{sx}+{xin}]"
                )
                + (
                    f"*p1[(z*{experts}+e)*{numel}+{sw}+v*{ci}+u];"
                    if transposed
                    else f"*p1[(z*{experts}+e)*{numel}+{sw}+u*{co}+v];"
                ),
                "}",
                f"total += scalar({scale:.17g})*value;",
                "}",
            ]
        dim = entries[0][5]
        channels = entries[0][3 if role == 0 else 4]
        position = (
            f"(e*{channels}+c%{channels})*{dim}+c/{channels}"
            if layout == "mul_ir"
            else f"(c/{channels}*{experts}+e)*{channels}+c%{channels}"
        )
        if role == 0 and not transposed:
            lines += [
                "for(int delta=16;delta>0;delta/=2) total+=__shfl_down_sync(0xffffffff,total,delta);",
                "if(lane!=0) return;",
            ]
        lines += [
            f"result[n*{din if role == 0 else dout}+{offset}+{position}]=total;",
            "return;",
            "}",
        ]
        cases.append("\n".join(lines))
    scalar = "double" if dtype == torch.float64 else "float"
    code = f"""
    using scalar={scalar};
    extern "C" __global__ void run(const scalar* __restrict__ p0,
        const scalar* __restrict__ p1, const scalar* __restrict__ p2,
        const long long* __restrict__ types, const long long* __restrict__ order,
        scalar* __restrict__ result, long long nodes) {{
        long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        int lane=threadIdx.x%32;
        {"index/=32;" if role == 0 and not transposed else ""}
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}
    """
    return (
        code,
        max(widths, default=0),
        len(groups),
        sum(widths) == (din if role == 0 else dout),
    )


def contract(metadata, role, node_type, features, weight, output):
    """Contract into features, coefficients or outputs without expanded weights."""
    result = linear_output(metadata, role, node_type, features, weight, output)
    if result.dtype not in (torch.float32, torch.float64):
        raise TypeError("Linear CUDA kernels support float32 and float64.")
    plan = parse_metadata(metadata)
    if not node_type.numel() or not plan[4]:
        return result.zero_()
    operands = [value.contiguous() for value in (features, weight, output)]
    # Transpose small element tables, not sample-wise matrices. Large tables
    # use warp reductions to avoid an additional table-sized allocation.
    transposed = role == 0 and weight.numel() * weight.element_size() <= 16 << 20
    if transposed:
        parts = weight.split([ci * co for _, _, _, ci, co, _, _ in plan[4]], -1)
        operands[1] = torch.cat(
            [
                part.reshape(*weight.shape[:2], ci, co).transpose(-1, -2).flatten(-2)
                for part, (_, _, _, ci, co, _, _) in zip(parts, plan[4])
            ],
            -1,
        ).contiguous()
    node_type = node_type.contiguous()
    order = node_type.argsort() if role == 1 else node_type
    if role == 1:
        code, width, groups, tile_rows = weight_source(plan, result.dtype)
        blocks = ((node_type.numel() + tile_rows - 1) // tile_rows) * width
        threads, complete = 256, False
    else:
        code, width, groups, complete = source(plan, role, result.dtype, transposed)
        lanes = 32 if role == 0 and not transposed else 1
        blocks = (node_type.numel() * width * lanes + 127) // 128
        threads = 128
    if not complete:
        result.zero_()
    kernel = kernels((code,), result.device)[code]
    args = [value.data_ptr() for value in operands] + [
        node_type.data_ptr(),
        order.data_ptr(),
        result.data_ptr(),
        node_type.numel(),
    ]
    runtime().launch(
        [(kernel, args, blocks, groups, threads, 0)],
        torch.cuda.current_stream(result.device).cuda_stream,
    )
    return result


@triton.jit
def map_kernel(
    features,
    weight,
    output,
    types,
    result,
    nodes,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    PATHS: tl.constexpr,
    ROLE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    OFFSET: tl.constexpr,
    CHANNELS: tl.constexpr,
    DIM: tl.constexpr,
    MUL_IR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    node = index // (E * CHANNELS * DIM)
    expert = (index // (CHANNELS * DIM)) % E
    component = (index // CHANNELS) % DIM
    channel = index % CHANNELS
    element = tl.load(types + node, node < nodes, 0)
    total = tl.full((BLOCK,), 0, result.dtype.element_ty)
    for path in tl.static_range(len(PATHS)):
        sx, sw, sy, ci, co, dim, scale = PATHS[path]
        if ROLE == 2:
            if sy == OFFSET and co == CHANNELS and dim == DIM:
                value = tl.full((BLOCK,), 0, result.dtype.element_ty)
                for k in range(ci):
                    if MUL_IR:
                        address = node * DIN + sx + (expert * ci + k) * dim + component
                    else:
                        address = node * DIN + sx + (component * E + expert) * ci + k
                    x = tl.load(features + address, node < nodes, 0)
                    w = tl.load(
                        weight + (element * E + expert) * W + sw + k * co + channel,
                        node < nodes,
                        0,
                    )
                    value = tl.fma(x, w, value)
                total += value * tl.full((), PATHS[path][6], result.dtype.element_ty)
        else:
            if sx == OFFSET and ci == CHANNELS and dim == DIM:
                value = tl.full((BLOCK,), 0, result.dtype.element_ty)
                for k in range(co):
                    if MUL_IR:
                        address = node * DOUT + sy + (expert * co + k) * dim + component
                    else:
                        address = node * DOUT + sy + (component * E + expert) * co + k
                    y = tl.load(output + address, node < nodes, 0)
                    windex = k * ci + channel if TRANSPOSED else channel * co + k
                    w = tl.load(
                        weight + (element * E + expert) * W + sw + windex,
                        node < nodes,
                        0,
                    )
                    value = tl.fma(y, w, value)
                total += value * tl.full((), PATHS[path][6], result.dtype.element_ty)
    width = DIN if ROLE == 0 else DOUT
    if MUL_IR:
        address = (
            node * width + OFFSET + (expert * CHANNELS + channel) * DIM + component
        )
    else:
        address = node * width + OFFSET + (component * E + expert) * CHANNELS + channel
    tl.store(result + address, total, node < nodes)


@triton.jit
def weight_kernel(
    features,
    output,
    order,
    starts,
    ends,
    result,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    SY: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    DIM: tl.constexpr,
    SCALE: tl.constexpr,
    MUL_IR: tl.constexpr,
    BI: tl.constexpr,
    BO: tl.constexpr,
):
    element, expert = tl.program_id(0) // E, tl.program_id(0) % E
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    o = tl.program_id(2) * BO + tl.arange(0, BO)
    first, last = tl.load(starts + element), tl.load(ends + element)
    total = tl.full((BI, BO), 0, result.dtype.element_ty)
    for row in range(first, last):
        node = tl.load(order + row)
        for m in range(DIM):
            if MUL_IR:
                xi = node * DIN + SX + (expert * CI + i) * DIM + m
                yo = node * DOUT + SY + (expert * CO + o) * DIM + m
            else:
                xi = node * DIN + SX + (m * E + expert) * CI + i
                yo = node * DOUT + SY + (m * E + expert) * CO + o
            x = tl.load(features + xi, i < CI, 0)
            y = tl.load(output + yo, o < CO, 0)
            total = tl.fma(x[:, None], y[None, :], total)
    address = (element * E + expert) * W + SW + i[:, None] * CO + o[None, :]
    mask = (i[:, None] < CI) & (o[None, :] < CO)
    # Separate paths may share coefficient ranges. Launches serialize on stream.
    previous = tl.load(result + address, mask, 0)
    tl.store(
        result + address,
        previous + total * tl.full((), SCALE, result.dtype.element_ty),
        mask,
    )


@triton.jit
def input_kernel(
    output,
    weight,
    types,
    result,
    nodes,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    PATHS: tl.constexpr,
    OFFSET: tl.constexpr,
    CHANNELS: tl.constexpr,
    DIM: tl.constexpr,
    MUL_IR: tl.constexpr,
    ROWS: tl.constexpr,
):
    index = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 32)
    node = index // (E * CHANNELS * DIM)
    expert = (index // (CHANNELS * DIM)) % E
    component = (index // CHANNELS) % DIM
    channel = index % CHANNELS
    element = tl.load(types + node, node < nodes, 0)
    total = tl.full((ROWS,), 0, result.dtype.element_ty)
    for path in tl.static_range(len(PATHS)):
        sx, sw, sy, ci, co, dim, scale = PATHS[path]
        partial = tl.full((ROWS, 32), 0, result.dtype.element_ty)
        for block in range((co + 31) // 32):
            k = block * 32 + lane
            if MUL_IR:
                address = (
                    node[:, None] * DOUT
                    + sy
                    + (expert[:, None] * co + k[None, :]) * dim
                    + component[:, None]
                )
            else:
                address = (
                    node[:, None] * DOUT
                    + sy
                    + (component[:, None] * E + expert[:, None]) * co
                    + k[None, :]
                )
            mask = (node[:, None] < nodes) & (k[None, :] < co)
            y = tl.load(output + address, mask, 0)
            w = tl.load(
                weight
                + (element[:, None] * E + expert[:, None]) * W
                + sw
                + channel[:, None] * co
                + k[None, :],
                mask,
                0,
            )
            partial = tl.fma(y, w, partial)
        total += tl.sum(partial, axis=1) * tl.full(
            (), PATHS[path][6], result.dtype.element_ty
        )
    if MUL_IR:
        address = node * DIN + OFFSET + (expert * CHANNELS + channel) * DIM + component
    else:
        address = node * DIN + OFFSET + (component * E + expert) * CHANNELS + channel
    tl.store(result + address, total, node < nodes)


@triton_op("eqx::linear_compiled", mutates_args=())
def compiled_contract(
    metadata: str,
    role: int,
    node_type: torch.Tensor,
    features: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Use FP32/FP64 FMA and element tables for all three contraction roles."""
    validate_uniform((features, weight, output))
    din, dout, numel, experts, paths, layout = parse_metadata(metadata)
    features, weight, output = (
        value.contiguous() for value in (features, weight, output)
    )
    # Keep metadata with gaps/repeated coefficient ranges on the general path.
    sizes = [ci * co for _, _, _, ci, co, _, _ in paths]
    partitioned = (
        bool(paths)
        and sum(sizes) == numel
        and all(path[1] == sum(sizes[:i]) for i, path in enumerate(paths))
    )
    transposed = (
        role == 0 and partitioned and weight.numel() * weight.element_size() <= 16 << 20
    )
    if transposed:
        parts = weight.split(sizes, dim=-1)
        weight = torch.cat(
            [
                part.reshape(*weight.shape[:2], path[3], path[4])
                .transpose(-1, -2)
                .flatten(-2)
                for part, path in zip(parts, paths)
            ],
            dim=-1,
        ).contiguous()
    types = node_type.contiguous()
    result = linear_output(metadata, role, types, features, weight, output).zero_()
    if role == 1:
        sorted_types, order = types.sort()
        ends = torch.searchsorted(
            sorted_types, torch.arange(1, weight.shape[0] + 1, device=types.device)
        )
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        for sx, sw, sy, ci, co, dim, scale in paths:
            wrap_triton(weight_kernel)[
                (weight.shape[0] * experts, triton.cdiv(ci, 8), triton.cdiv(co, 32))
            ](
                features,
                output,
                order,
                starts,
                ends,
                result,
                din,
                dout,
                numel,
                experts,
                sx,
                sw,
                sy,
                ci,
                co,
                dim,
                scale,
                layout == "mul_ir",
                8,
                32,
                num_warps=4,
            )
    else:
        groups = defaultdict(list)
        for path in paths:
            sx, sw, sy, ci, co, dim, scale = path
            groups[(sx, ci, dim) if role == 0 else (sy, co, dim)].append(path)
        for (offset, channels, dim), entries in groups.items():
            if role == 0 and weight.numel() * weight.element_size() > 16 << 20:
                wrap_triton(input_kernel)[
                    (triton.cdiv(types.shape[0] * experts * channels * dim, 4),)
                ](
                    output,
                    weight,
                    types,
                    result,
                    types.shape[0],
                    din,
                    dout,
                    numel,
                    experts,
                    tuple(entries),
                    offset,
                    channels,
                    dim,
                    layout == "mul_ir",
                    4,
                    num_warps=4,
                )
                continue
            wrap_triton(map_kernel)[
                (triton.cdiv(types.shape[0] * experts * channels * dim, 128),)
            ](
                features,
                weight,
                output,
                types,
                result,
                types.shape[0],
                din,
                dout,
                numel,
                experts,
                tuple(entries),
                role,
                transposed,
                offset,
                channels,
                dim,
                layout == "mul_ir",
                128,
                num_warps=4,
            )
    return result


compiled_contract.register_autograd(backward, setup_context=setup_context)
