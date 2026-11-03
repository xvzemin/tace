"""CUDA indexed channel contractions and their transposes."""

from collections import defaultdict
from functools import lru_cache

import torch

from ...utils.metadata import parse_metadata
from ...utils.outputs import linear_output
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
