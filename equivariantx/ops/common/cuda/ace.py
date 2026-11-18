"""Specialized sparse ACE kernels and transposed contractions."""

from collections import Counter, defaultdict
from functools import lru_cache

import torch

from ....utils.metadata import parse_metadata
from ....utils.outputs import ace_outputs
from .runtime import kernels, runtime


@lru_cache(maxsize=128)
def schedule_batches(groups, max_entries=1024):
    """Partition contraction schedules into bounded CUDA translation units."""
    batches, batch, size = [], [], 0
    for key, entries in groups:
        for start in range(0, len(entries), max_entries):
            values = entries[start : start + max_entries]
            if batch and size + len(values) > max_entries:
                batches.append(tuple(batch))
                batch, size = [], 0
            batch.append((key, values))
            size += len(values)
    if batch:
        batches.append(tuple(batch))
    return tuple(batches)


def single_writer(blocks, size, *, complete=True):
    """Check that writes are disjoint, optionally covering the whole destination."""
    if not size or size > 1 << 18:
        return False
    written = bytearray(size)
    for start, channels, stride, span in blocks:
        for channel in range(channels):
            lo = start + channel * stride
            if lo < 0 or lo + span > size or any(written[lo : lo + span]):
                return False
            written[lo : lo + span] = b"\1" * span
    return not complete or all(written)


@lru_cache(maxsize=128)
def weighted_source(dtype, dims, schedules, roles, unique):
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
    direct = []
    for role in unique:
        if role == 2:
            continue
        blocks = []
        for (_, ci, co, dx, dy, dz), entries in paths.items():
            offsets = [r[role] for r, _ in entries]
            blocks.append(
                (
                    min(offsets),
                    co if role == 3 else ci,
                    {0: dx, 1: dy, 3: dz}[role],
                    max(offsets) - min(offsets) + 1,
                )
            )
        if single_writer(blocks, dims[role]):
            direct.append(role)

    def store(role, index, value):
        return (
            f"o{role}[{index}]={value};"
            if role in direct
            else f"atomicAdd(o{role}+{index},{value});"
        )

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
            for r, coefficient in entries:
                parts.append(
                    f"a[{r[3] - so}]+=scalar({coefficient:.17g})*"
                    f"p0[n*{dims[0]}+{r[0]}+u*{dx}]*"
                    f"p1[n*{dims[1]}+{r[1]}+u*{dy}];"
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
                + store(3, f"(n0+row)*{dims[3]}+{so}+v*{dz}+m", "sum[row]"),
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
                f"for(int u=t/32;u<{ci};u+=blockDim.x/32) {{",
                f"scalar sum[{nz}]={{}};",
                f"for(int v=t%32;v<{co};v+=32) {{",
                f"scalar w=p2[types[n0]*{dims[2]}+{sw}+u*{co}+v];",
                f"#pragma unroll\nfor(int m=0;m<{nz};++m) "
                f"sum[m]+=w*p3[n0*{dims[3]}+{so}+v*{dz}+m]; }}",
                f"#pragma unroll\nfor(int m=0;m<{nz};++m) {{",
                "for(int s=16;s>0;s/=2) sum[m]+=__shfl_down_sync(0xffffffff,sum[m],s);",
                f"if(t%32==0) tile[u*{nz}+m]=sum[m]; }} }}",
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
                        f"for(int m=0;m<{size};++m) "
                        + store(
                            role,
                            f"n0*{dims[role]}+{offset}+u*{stride}+m",
                            f"g{role}[m]",
                        )
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
        tuple(direct),
    )


@lru_cache(maxsize=128)
def source(dtype, dims, weighted, role, groups, write):
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
            else f"atomicAdd(out+n*{dims[role]}+{offset}+c*{stride},total);"
            if write == "atomic"
            else f"out[n*{dims[role]}+{offset}+c*{stride}] {'=' if write == 'assign' else '+='} total;"
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
def launch_plan(metadata, program, dtype):
    """Schedule immutable path and adjoint instructions independently of node count."""
    dims, weighted, schedules = parse_metadata(metadata)
    grouped = defaultdict(list)
    read = {0}
    terms = parse_metadata(program)
    writers = Counter(slot for _, _, slot in terms)
    initialized = set()
    for mapping, role, slot in terms:
        grouped[mapping, 0 if role in (0, 1) else role].append((role, slot))
        read.update(1 + index for i, index in enumerate(mapping) if i != role)
    jobs = []
    for (mapping, _), terms in grouped.items():
        roles = tuple(sorted(role for role, _ in terms))
        # Mixed higher derivatives can contribute more than once to a role.
        tiled = (
            weighted_source(
                dtype,
                dims,
                schedules,
                roles,
                tuple(r for r, slot in terms if writers[slot] == 1),
            )
            if weighted
            and sum(len(entries) for _, entries in schedules[3]) <= 2048
            and len(set(roles)) == len(roles)
            and (all(role in (0, 1) for role in roles) or len(roles) == 1)
            else None
        )
        if tiled is not None:
            code, count, rows, direct = tiled
            destinations = dict(terms)
            initialized.update(destinations[r] for r in direct)
            jobs.append(
                (code, mapping, tuple(destinations[r] for r in roles), 1, rows, count)
            )
            continue
        for role, slot in terms:
            groups = schedules[role]
            if not groups:
                continue
            batches = schedule_batches(groups)
            direct = (
                writers[slot] == 1
                and not (weighted and role == 2)
                and single_writer(
                    ((*key, 1) for batch in batches for key, _ in batch), dims[role]
                )
            )
            if direct:
                initialized.add(slot)
            for batch in batches:
                # Reblocked input multiplicities can partially overlap even
                # when the schedule keys are different. Separate launches are ordered.
                write = "atomic"
                if direct:
                    write = "assign"
                elif not (weighted and role == 2) and single_writer(
                    ((*key, 1) for key, _ in batch), dims[role], complete=False
                ):
                    write = "add"
                code = source(dtype, dims, weighted, role, batch, write)
                width = max(key[1] for key, _ in batch)
                jobs.append(
                    (
                        code,
                        mapping + (-1,) * (4 - len(mapping)),
                        (slot,),
                        width,
                        128,
                        len(batch),
                    )
                )
    return tuple(jobs), frozenset(read), frozenset(initialized)


def launch(jobs, read, initialized, node_type, operands, outputs):
    """Bind tensors to the cached schedule on the current CUDA stream."""
    nodes = node_type.numel()
    for slot, output in enumerate(outputs):
        if not nodes or slot not in initialized:
            output.zero_()
    if not nodes:
        return
    device = node_type.device
    operands = [
        value.contiguous() if i + 1 in read else value
        for i, value in enumerate(operands)
    ]
    node_type = node_type.contiguous()
    pointers = [value.data_ptr() for value in operands] + [0]
    destinations = [value.data_ptr() for value in outputs]
    compiled = kernels([job[0] for job in jobs], device)
    launches = []
    for code, mapping, slots, width, rows, ny in jobs:
        args = [pointers[i] for i in mapping] + [node_type.data_ptr()]
        args.extend(destinations[slot] for slot in slots)
        args.append(nodes)
        launches.append(
            (compiled[code], args, (nodes * width + rows - 1) // rows, ny, 128, 0)
        )
    runtime().launch(launches, torch.cuda.current_stream(device).cuda_stream)


def contract(
    metadata: str,
    program: str,
    node_type: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate a multilinear contraction without expanded CG products."""
    from .graph import execute

    if operands[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("ACE CUDA kernels support float32 and float64.")
    outputs = ace_outputs(metadata, program, node_type, operands)
    jobs, read, initialized = launch_plan(metadata, program, operands[0].dtype)

    def run(inputs, outputs):
        launch(jobs, read, initialized, inputs[0], inputs[1:], outputs)

    execute("ace", (metadata, program), run, (node_type, *operands), outputs, read=read)
    return outputs
