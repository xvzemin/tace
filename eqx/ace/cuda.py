"""Specialized sparse ACE kernels and transposed contractions."""

from collections import defaultdict
from functools import lru_cache

import torch

from ..kernels.codegen import contraction_source
from ..kernels.cuda import kernels, runtime
from ..utils.metadata import parse_metadata


def bilinear_terms(terms, factor=False):
    """Reuse input products across angular output components."""
    if factor:
        outputs = defaultdict(list)
        for destination, coefficient, left, right in terms:
            outputs[destination].append((coefficient, (left, right)))
        lines, cache = [], {}
        for destination, entries in outputs.items():
            value = contraction_source(entries, cache, lines)
            lines.append(f"{destination}+={value};")
        return lines
    products = defaultdict(list)
    for destination, coefficient, left, right in terms:
        products[left, right].append((destination, coefficient))
    lines = []
    for (left, right), outputs in products.items():
        lines.append(f"{{ scalar product={left}*{right};")
        for destination, coefficient in outputs:
            lines.append(f"{destination}+=scalar({coefficient:.17g})*product;")
        lines.append("}")
    return lines


@lru_cache(maxsize=128)
def component_source(dtype, dims, role, groups):
    """Evaluate all independent components of a Cartesian output together."""
    blocks = defaultdict(list)
    for _, entries in groups:
        for r, coefficient in entries:
            if len(r) != 12:
                return None
            blocks[r[9 + role], r[4], r[6 + role]].append((r, coefficient))
    if not blocks:
        return None
    scalar = "double" if dtype == torch.float64 else "float"
    cases = []
    for index, ((offset, channels, stride), entries) in enumerate(blocks.items()):
        lines = [
            f"case {index}: {{ if(index>=nodes*{channels}) return;",
            f"long long n=index/{channels}; int c=index%{channels};",
            f"scalar values[{stride}]={{}};",
        ]
        terms = []
        for r, coefficient in entries:
            operands = [
                f"p{i}[n*{dims[i]}+{r[i]}+c*{r[6 + i]}]" for i in range(3) if i != role
            ]
            terms.append((f"values[{r[role] - offset}]", coefficient, *operands))
        lines.extend(bilinear_terms(terms, factor=True))
        lines.append(
            f"for(int m=0;m<{stride};++m) "
            f"out[n*{dims[role]}+{offset}+c*{stride}+m]+=values[m]; return; }}"
        )
        cases.append("\n".join(lines))
    return (
        f"""using scalar={scalar}; using T=scalar;
        extern "C" __global__ void run(const scalar* p0, const scalar* p1,
            const scalar* p2, const scalar* p3, const long long* types,
            scalar* out, long long nodes) {{
            long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            switch(blockIdx.y) {{ {"".join(cases)} }}
        }}""",
        max(key[1] for key in blocks),
        len(blocks),
    )


@lru_cache(maxsize=128)
def weighted_source(dtype, dims, schedules, roles):
    """Share angular products and coefficient adjoints across channels."""
    paths = defaultdict(list)
    for _, entries in schedules[3]:
        for r, coefficient in entries:
            paths[(r[2], r[4], r[5], *r[6:9])].append((r, coefficient))
    if not paths:
        return None
    scalar = "double" if dtype == torch.float64 else "float"
    itemsize = 8 if dtype == torch.float64 else 4
    cases, storage = [], 1
    rows = 4 if roles in ((3,), (2,)) else 1
    for index, ((sw, ci, co, dx, dy, dz), entries) in enumerate(paths.items()):
        sx = min(r[0] for r, _ in entries)
        sy = min(r[1] for r, _ in entries)
        so = min(r[3] for r, _ in entries)
        nx = max(r[0] - sx for r, _ in entries) + 1
        ny = max(r[1] - sy for r, _ in entries) + 1
        nz = max(r[3] - so for r, _ in entries) + 1
        storage = max(storage, rows * ci * nz)
        parts = [f"case {index}: {{"]
        if roles in ((3,), (2,)):
            parts += [
                f"for(int q=t;q<{rows * ci};q+=blockDim.x) {{",
                f"int u=q%{ci}; long long n=n0+q/{ci}; scalar a[{nz}]={{}};",
                "if(n<nodes) {",
            ]
            parts.extend(
                bilinear_terms(
                    [
                        (
                            f"a[{r[3] - so}]",
                            coefficient,
                            f"p0[n*{dims[0]}+{r[0]}+u*{dx}]",
                            f"p1[n*{dims[1]}+{r[1]}+u*{dy}]",
                        )
                        for r, coefficient in entries
                    ]
                )
            )
            parts += [
                "}",
                f"for(int m=0;m<{nz};++m) tile[q*{nz}+m]=a[m]; }}",
                "__syncthreads();",
            ]
        if roles == (3,):
            parts += [
                f"for(int q=t;q<{co * nz};q+=blockDim.x) {{",
                f"int v=q/{nz}, m=q%{nz}; scalar sum[{rows}]={{}};",
                f"for(int u=0;u<{ci};++u) {{",
                f"#pragma unroll\nfor(int row=0;row<{rows};++row) if(n0+row<nodes) "
                f"sum[row]+=tile[(row*{ci}+u)*{nz}+m]*"
                f"p2[types[n0+row]*{dims[2]}+{sw}+u*{co}+v];",
                "}",
                f"#pragma unroll\nfor(int row=0;row<{rows};++row) if(n0+row<nodes) "
                f"atomicAdd(o3+(n0+row)*{dims[3]}+{so}+v*{dz}+m,sum[row]);",
                "}",
            ]
        elif roles == (2,):
            parts += [
                f"for(int q=t;q<{ci * co};q+=blockDim.x) {{",
                f"int u=q/{co}, v=q%{co};",
                f"#pragma unroll\nfor(int row=0;row<{rows};++row) {{",
                "if(n0+row>=nodes) continue; long long z=types[n0+row];",
                "bool first=true; for(int k=0;k<row;++k) if(types[n0+k]==z) first=false;",
                "if(!first) continue; scalar sum=0;",
                f"#pragma unroll\nfor(int k=row;k<{rows};++k) if(n0+k<nodes && types[n0+k]==z) {{",
                f"for(int m=0;m<{nz};++m) sum+=tile[(k*{ci}+u)*{nz}+m]*"
                f"p3[(n0+k)*{dims[3]}+{so}+v*{dz}+m];",
                "}",
                f"atomicAdd(o2+z*{dims[2]}+{sw}+q,sum);",
                "} }",
            ]
        else:
            parts += [
                f"for(int q=t/32;q<{ci * nz};q+=blockDim.x/32) {{",
                f"int u=q/{nz}, m=q%{nz}; scalar sum=0;",
                f"for(int v=t%32;v<{co};v+=32) sum+="
                f"p2[types[n0]*{dims[2]}+{sw}+u*{co}+v]*"
                f"p3[n0*{dims[3]}+{so}+v*{dz}+m];",
                "for(int s=16;s>0;s/=2) sum+=__shfl_down_sync(0xffffffff,sum,s);",
                "if(t%32==0) tile[q]=sum; }",
                "__syncthreads();",
                f"for(int u=t;u<{ci};u+=blockDim.x) {{",
            ]
            for role in roles:
                parts.append(f"scalar g{role}[{nx if role == 0 else ny}]={{}};")
            for r, coefficient in entries:
                parts.append(
                    f"{{ scalar a=scalar({coefficient:.17g})*tile[u*{nz}+{r[3] - so}];"
                )
                if 0 in roles:
                    parts.append(f"g0[{r[0] - sx}]+=a*p1[n0*{dims[1]}+{r[1]}+u*{dy}];")
                if 1 in roles:
                    parts.append(f"g1[{r[1] - sy}]+=a*p0[n0*{dims[0]}+{r[0]}+u*{dx}];")
                parts.append("}")
            for role, size, offset, stride in ((0, nx, sx, dx), (1, ny, sy, dy)):
                if role in roles:
                    parts.append(
                        f"for(int m=0;m<{size};++m) atomicAdd(o{role}+n0*{dims[role]}+{offset}+u*{stride}+m,g{role}[m]);"
                    )
            parts.append("}")
        cases.append("\n".join([*parts, "return; }"]))
    if storage * itemsize > 48 << 10:
        return None
    destinations = ", ".join(f"scalar* o{role}" for role in roles)
    return (
        f"""using scalar={scalar};
    extern "C" __global__ void run(const scalar* p0,const scalar* p1,
        const scalar* p2,const scalar* p3,const long long* types,
        {destinations},long long nodes) {{
        __shared__ scalar tile[{storage}];
        long long n0=(long long)blockIdx.x*{rows}; int t=threadIdx.x;
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}""",
        len(paths),
        rows,
    )


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


@lru_cache(maxsize=256)
def execution_plan(metadata, program, dtype, device):
    """Compile fixed contraction schedules independently of the node count."""
    dims, weighted, schedules = parse_metadata(metadata)
    grouped = defaultdict(list)
    for mapping, role, slot in parse_metadata(program):
        grouped[mapping, 0 if role in (0, 1) else role].append((role, slot))
    jobs = []
    for (mapping, _), terms in grouped.items():
        roles = tuple(sorted(role for role, _ in terms))
        # Mixed higher derivatives can contribute more than once to a role.
        tiled = (
            weighted_source(dtype, dims, schedules, roles)
            if weighted
            and len(set(roles)) == len(roles)
            and (all(role in (0, 1) for role in roles) or len(roles) == 1)
            else None
        )
        if tiled is not None:
            code, count, rows = tiled
            destinations = dict(terms)
            jobs.append(
                (
                    code,
                    mapping,
                    tuple(destinations[role] for role in roles),
                    rows,
                    1,
                    count,
                )
            )
            continue
        for role, slot in terms:
            groups = schedules[role]
            if not groups:
                continue
            compact = (
                component_source(dtype, dims, role, groups) if not weighted else None
            )
            if compact is None:
                code = source(dtype, dims, weighted, role, groups)
                width, count = max(key[1] for key, _ in groups), len(groups)
            else:
                code, width, count = compact
            jobs.append((code, mapping, (slot,), 128, width, count))
    compiled = kernels([job[0] for job in jobs], device)
    return tuple((compiled[code], *job) for code, *job in jobs)


def launch(metadata, program, node_type, operands, outputs):
    if operands[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("ACE CUDA kernels support float32 and float64.")
    operands = [value.contiguous() for value in operands]
    node_type = node_type.contiguous()
    for output in outputs:
        output.zero_()
    if not node_type.numel():
        return
    device = operands[0].device
    plan = execution_plan(metadata, program, operands[0].dtype, device)
    launches = []
    for kernel, mapping, slots, rows, width, count in plan:
        args = [operands[i].data_ptr() for i in mapping]
        args += [0] * (4 - len(args))
        args += [
            node_type.data_ptr(),
            *(outputs[slot].data_ptr() for slot in slots),
            node_type.numel(),
        ]
        launches.append(
            (
                kernel,
                args,
                (node_type.numel() * width + rows - 1) // rows,
                count,
                128,
                0,
            )
        )
    runtime().launch(launches, torch.cuda.current_stream(device).cuda_stream)
