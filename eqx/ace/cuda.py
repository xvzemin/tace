"""Specialized sparse ACE kernels and transposed contractions."""

from collections import defaultdict
from functools import lru_cache

import torch

from ..kernels.cuda import kernels, runtime


@lru_cache(maxsize=128)
def source(dtype, dims, weighted, role, groups):
    scalar = "double" if dtype == torch.float64 else "float"
    cases = []
    for group, ((offset, channels, stride), entries) in enumerate(groups):
        parts = [
            f"case {group}: {{",
            f"if(index>=nodes*{channels}) return;",
            f"long long n=index/{channels}; int c=index%{channels};",
            "long long z=types[n]; scalar total=0;",
        ]
        if weighted:
            # Factor the dense coefficient out of the angular contraction.
            # A path is still evaluated independently; no basis compression.
            factored = defaultdict(list)
            for r, coefficient in entries:
                factored[r[2], r[4], r[5]].append((r, coefficient))
            for (weight_offset, ci_size, co_size), terms in factored.items():
                channel = (
                    "int ci=c, co=other;"
                    if role in (0, 1)
                    else f"int ci=c/{co_size}, co=c%{co_size};"
                    if role == 2
                    else "int ci=other, co=c;"
                )
                width = 1 if role == 2 else co_size if role in (0, 1) else ci_size
                parts += [
                    "{",
                    f"for(int other=0;other<{width};++other) {{",
                    channel,
                    "scalar angular=0;",
                ]
                for r, coefficient in terms:
                    addresses = {
                        0: f"n*{dims[0]}+{r[0]}+ci*{r[6]}",
                        1: f"n*{dims[1]}+{r[1]}+ci*{r[7]}",
                        3: f"n*{dims[3]}+{r[3]}+co*{r[8]}",
                    }
                    product = " * ".join(
                        f"p{i}[{address}]"
                        for i, address in addresses.items()
                        if i != role
                    )
                    parts.append(
                        f"angular += scalar({coefficient:.17g}) * ({product});"
                    )
                parts.append(
                    "total += angular;"
                    if role == 2
                    else f"total += angular*p2[z*{dims[2]}+{weight_offset}+ci*{co_size}+co];"
                )
                parts += ["}", "}"]
        else:
            for r, coefficient in entries:
                product = " * ".join(
                    f"p{i}[n*{dims[i]}+{r[i]}+c*{r[i + 6]}]"
                    for i in range(3)
                    if i != role
                )
                parts.append(f"total += scalar({coefficient:.17g}) * ({product});")
        parts.append(
            f"atomicAdd(out+z*{dims[role]}+{offset}+c,total);"
            if weighted and role == 2
            else f"out[n*{dims[role]}+{offset}+c*{stride}] += total;"
        )
        parts += ["return;", "}"]
        cases.append("\n".join(parts))
    return f"""
    using scalar={scalar};
    extern "C" __global__ void run(const scalar* p0, const scalar* p1,
        const scalar* p2, const scalar* p3, const long long* types,
        scalar* out, long long nodes) {{
        long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}
    """


def launch(metadata, program, node_type, operands, outputs):
    dims, weighted, schedules = metadata
    if operands[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("ACE CUDA kernels support float32 and float64.")
    operands = [value.contiguous() for value in operands]
    node_type = node_type.contiguous()
    for output in outputs:
        output.zero_()
    if not node_type.numel():
        return
    device = operands[0].device
    codes = {
        role: source(operands[0].dtype, dims, weighted, role, schedules[role])
        for _, role, _ in program
    }
    compiled = kernels(codes.values(), device)
    launches = []
    for mapping, role, slot in program:
        groups = schedules[role]
        if not groups:
            continue
        width = max(key[1] for key, _ in groups)
        pointers = [operands[i].data_ptr() for i in mapping]
        pointers += [0] * (4 - len(pointers))
        args = pointers + [
            node_type.data_ptr(),
            outputs[slot].data_ptr(),
            node_type.numel(),
        ]
        launches.append(
            (
                compiled[codes[role]],
                args,
                (node_type.numel() * width + 127) // 128,
                len(groups),
                128,
                0,
            )
        )
    runtime().launch(launches, torch.cuda.current_stream(device).cuda_stream)
