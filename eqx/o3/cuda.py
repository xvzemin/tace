"""CUDA indexed linear contractions without per-node weight tensors."""

from collections import defaultdict
from functools import lru_cache

import torch

from ..kernels.cuda import kernels, runtime


@lru_cache(maxsize=256)
def source(metadata, role, dtype):
    din, dout, numel, experts, paths = metadata
    groups = defaultdict(list)
    for path in paths:
        sx, sw, sy, ci, co, dim, scale = path
        key = (
            (sw, ci * co)
            if role == 1
            else (sx, ci * dim)
            if role == 0
            else (sy, co * dim)
        )
        groups[key].append(path)
    cases = []
    widths = []
    for group, ((offset, width), entries) in enumerate(groups.items()):
        widths.append(experts * width)
        lines = [
            f"case {group}: {{",
            f"if(index>=nodes*{experts * width}) return;",
            f"long long n=index/{experts * width}; int e=(index/{width})%{experts};",
            f"int c=index%{width}; long long z=types[n]; scalar total=0;",
        ]
        for sx, sw, sy, ci, co, dim, scale in entries:
            if role == 1:
                lines += [
                    f"int u=c/{co}, v=c%{co};",
                    "scalar angular=0;",
                    f"for(int m=0;m<{dim};++m) angular += p0[n*{din}+{sx}+(e*{ci}+u)*{dim}+m]*p2[n*{dout}+{sy}+(e*{co}+v)*{dim}+m];",
                    f"total += scalar({scale:.17g})*angular;",
                ]
            else:
                channel = (
                    f"int u=c/{dim}, m=c%{dim};"
                    if role == 0
                    else f"int v=c/{dim}, m=c%{dim};"
                )
                loop = (
                    f"for(int v=0;v<{co};++v)"
                    if role == 0
                    else f"for(int u=0;u<{ci};++u)"
                )
                feature = (
                    f"p2[n*{dout}+{sy}+(e*{co}+v)*{dim}+m]"
                    if role == 0
                    else f"p0[n*{din}+{sx}+(e*{ci}+u)*{dim}+m]"
                )
                lines += [
                    "{",
                    channel,
                    "scalar value=0;",
                    loop + " {",
                    f"value += {feature}*p1[(z*{experts}+e)*{numel}+{sw}+u*{co}+v];",
                    "}",
                    f"total += scalar({scale:.17g})*value;",
                    "}",
                ]
        lines += (
            [f"atomicAdd(result+(z*{experts}+e)*{numel}+{offset}+c,total);"]
            if role == 1
            else [f"result[n*{din if role == 0 else dout}+{offset}+e*{width}+c]=total;"]
        )
        lines += ["return;", "}"]
        cases.append("\n".join(lines))
    scalar = "double" if dtype == torch.float64 else "float"
    code = f"""
    using scalar={scalar};
    extern "C" __global__ void run(const scalar* p0, const scalar* p1,
        const scalar* p2, const long long* types, scalar* result, long long nodes) {{
        long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}
    """
    return code, max(widths, default=0), len(groups)


def launch(metadata, role, node_type, operands, result):
    if result.dtype not in (torch.float32, torch.float64):
        raise TypeError("Linear CUDA kernels support float32 and float64.")
    result.zero_()
    if not node_type.numel() or not metadata[-1]:
        return
    operands = [value.contiguous() for value in operands]
    node_type = node_type.contiguous()
    code, width, groups = source(metadata, role, result.dtype)
    kernel = kernels((code,), result.device)[code]
    args = [value.data_ptr() for value in operands] + [
        node_type.data_ptr(),
        result.data_ptr(),
        node_type.numel(),
    ]
    runtime().launch(
        [(kernel, args, (node_type.numel() * width + 127) // 128, groups, 128, 0)],
        torch.cuda.current_stream(result.device).cuda_stream,
    )
