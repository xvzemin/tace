"""Block-local bilinear products and factored transposed contractions."""

from collections import defaultdict
from functools import lru_cache

import torch

from ...kernels.cuda import kernels, runtime
from .bilinear_contraction import decode_metadata


@lru_cache(maxsize=128)
def forward_source(dtype, metadata, terms):
    plans = decode_metadata(metadata)
    groups = defaultdict(list)
    for term_index, (plan, mapping, _, _) in enumerate(terms):
        for _, entries in plans[plan][2][3]:
            for r, _ in entries:
                groups[r[10], r[18]].append((plan, mapping, r, term_index))
    cases = []
    shared_size = 1
    for index, ((gate, _), entries) in enumerate(sorted(groups.items())):
        angular, maps = {}, {}
        width = 0
        for plan, mapping, r, term_index in entries:
            shift = r[9] - gate
            width = max(width, shift + r[4])
            angular[r[0] - shift * r[6], r[1] - shift * r[7], r[6], r[7]] = r[12]
            maps[plan, mapping, r[2], r[3], shift, r[4], r[5], r[8], term_index] = r[11]
        dim = plans[entries[0][0]][0]
        mapping = entries[0][1]
        shared_size = max(shared_size, 4 * width)
        parts = [
            f"case {index}: {{",
            f"for(int q=t;q<{4 * width};q+=blockDim.x) {{ int u=q%{width}; long long n=n0+q/{width}; scalar v=0; if(n<nodes) {{",
        ]
        for (sx, sy, dx, dy), coefficient in angular.items():
            parts.append(
                f"v += scalar({coefficient:.17g})*p{mapping[0]}[n*{dim[0]}+{sx}+u*{dx}]*p{mapping[1]}[n*{dim[1]}+{sy}+u*{dy}];"
            )
        parts += [
            f"v*=p{mapping[4]}[n*{dim[4]}+{gate}+u]; }} a[q]=v; }}",
            "__syncthreads();",
        ]
        # Schedule expert outputs together. A narrow expert must not leave
        # the other warps idle while all experts execute sequentially.
        output_width = sum(key[6] for key in maps)
        parts.append(f"for(int task=t;task<{output_width};task+=blockDim.x) {{")
        output_offset = 0
        for (plan, mapping, sw, so, shift, ci, co, dout, _), scale in maps.items():
            dims = plans[plan][0]
            parts += [
                f"if(task>={output_offset} && task<{output_offset + co}) {{ int v=task-{output_offset}; scalar sum[4]={{}};",
                f"#pragma unroll 1\nfor(int u=0;u<{ci};++u) {{",
            ]
            if plan:
                parts += [
                    f"scalar w=p{mapping[2]}[{sw}+u*{co}+v];",
                    f"#pragma unroll\nfor(int row=0;row<4;++row) sum[row]+=a[row*{width}+{shift}+u]*w;",
                ]
            else:
                parts.append(
                    f"#pragma unroll\nfor(int row=0;row<4;++row) if(n0+row<nodes) sum[row]+=a[row*{width}+{shift}+u]*p{mapping[2]}[types[n0+row]*{dims[2]}+{sw}+u*{co}+v];"
                )
            parts += [
                "}",
                f"#pragma unroll\nfor(int row=0;row<4;++row) if(n0+row<nodes) atomicAdd(out+(n0+row)*{dims[3]}+{so}+v*{dout},scalar({scale:.17g})*sum[row]); }}",
            ]
            output_offset += co
        parts += ["}", "return; }"]
        cases.append("\n".join(parts))
    count = max(i for _, mapping, _, _ in terms for i in mapping) + 1
    pointers = ", ".join(f"const scalar* p{i}" for i in range(count))
    scalar = "double" if dtype == torch.float64 else "float"
    code = f"""using scalar={scalar};
    extern "C" __global__ void run({pointers}, const long long* types, scalar* out, long long nodes) {{
        __shared__ scalar a[{shared_size}];
        long long n0=(long long)blockIdx.x*4; int t=threadIdx.x;
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}"""
    return code, len(groups), count


@lru_cache(maxsize=128)
def transpose_source(dtype, metadata, terms):
    plans = decode_metadata(metadata)
    # Contract coefficients with output cotangents before sparse CG entries.
    # Angular products are independent of the output-channel summation.
    groups = defaultdict(list)
    role = terms[0][2]
    for term_index, (plan, mapping, _, _) in enumerate(terms):
        for key, entries in plans[plan][2][role]:
            groups[key].extend((plan, mapping, r, term_index) for r, _ in entries)
    cases = []
    for index, ((offset, channels, stride), entries) in enumerate(
        sorted(groups.items())
    ):
        parts = [
            f"case {index}: {{",
            f"if(index>=nodes*{channels}) return;",
            f"long long n=index/{channels}; int c=index%{channels}; scalar total=0;",
        ]
        factored = defaultdict(dict)
        for plan, mapping, r, term_index in entries:
            factored[plan, mapping, r[2], r[3], r[5], r[8], r[11], term_index][
                r[0], r[1], r[9], r[6], r[7]
            ] = r[12]
        for (plan, mapping, sw, so, co, dout, scale, _), angular in factored.items():
            dims = plans[plan][0]
            z = "0" if plan else "types[n]"
            parts += [
                "{ scalar adjoint=0;",
                f"for(int v=0;v<{co};++v) adjoint += p{mapping[2]}[{z}*{dims[2]}+{sw}+c*{co}+v]*p{mapping[3]}[n*{dims[3]}+{so}+v*{dout}];",
                "scalar angular=0;",
            ]
            for (sx, sy, sg, dx, dy), coefficient in angular.items():
                addresses = {
                    0: f"n*{dims[0]}+{sx}+c*{dx}",
                    1: f"n*{dims[1]}+{sy}+c*{dy}",
                    4: f"n*{dims[4]}+{sg}+c",
                }
                product = "*".join(
                    f"p{mapping[r]}[{address}]"
                    for r, address in addresses.items()
                    if r != role
                )
                parts.append(f"angular += scalar({coefficient:.17g})*({product});")
            parts += [f"total += scalar({scale:.17g})*adjoint*angular; }}"]
        dims = plans[entries[0][0]][0]
        parts.append(
            f"atomicAdd(out+n*{dims[role]}+{offset}+c*{stride},total); return; }}"
        )
        cases.append("\n".join(parts))
    count = max(i for _, mapping, _, _ in terms for i in mapping) + 1
    pointers = ", ".join(f"const scalar* p{i}" for i in range(count))
    scalar = "double" if dtype == torch.float64 else "float"
    code = f"""using scalar={scalar};
    extern "C" __global__ void run({pointers}, const long long* types, scalar* out, long long nodes) {{
      long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
      switch(blockIdx.y) {{ {"".join(cases)} }}
    }}"""
    return code, len(groups), max((key[1] for key in groups), default=0), count


@lru_cache(maxsize=128)
def inputs_source(dtype, metadata, branches, roles):
    plans = decode_metadata(metadata)
    paths = defaultdict(list)
    for branch, (plan, mapping) in enumerate(branches):
        for _, entries in plans[plan][2][3]:
            for r, _ in entries:
                paths[r[10]].append((plan, mapping, r, branch))
    cases = []
    shared_size = 1
    for index, (gate, entries) in enumerate(sorted(paths.items())):
        angular, maps = {}, {}
        width = 0
        for plan, mapping, r, branch in entries:
            shift = r[9] - gate
            width = max(width, shift + r[4])
            dx, dy, dz = r[6:9]
            a, b, c = r[16:19]
            sx, sy = r[13:15]
            angular[a, b, c] = r[12]
            maps[plan, mapping, r[2], r[3] - c, shift, r[4], r[5], branch] = r[11]
        dims = plans[entries[0][0]][0]
        mapping = entries[0][1]
        shared_size = max(shared_size, width * dz)
        parts = [
            f"case {index}: {{",
            f"for(int q=t/32;q<{width * dz};q+=blockDim.x/32) {{ int u=q/{dz}, m=q%{dz}; scalar sum=0;",
        ]
        for (plan, indices, sw, so, shift, ci, co, _), scale in maps.items():
            z = "0" if plan else "types[n]"
            parts += [
                f"if(u>={shift} && u<{shift + ci}) {{ scalar part=0;",
                f"for(int v=t%32;v<{co};v+=32) part+=p{indices[2]}[{z}*{plans[plan][0][2]}+{sw}+(u-{shift})*{co}+v]*p{indices[3]}[n*{dims[3]}+{so}+v*{dz}+m];",
                f"sum+=scalar({scale:.17g})*part; }}",
            ]
        parts += [
            "for(int s=16;s>0;s/=2) sum+=__shfl_down_sync(0xffffffff,sum,s);",
            "if(t%32==0) adj[q]=sum; }",
            "__syncthreads();",
            f"for(int u=t;u<{width};u+=blockDim.x) {{",
            f"scalar gx[{dx}]={{}}, gy[{dy}]={{}}, gg=0;",
            f"scalar gate=p{mapping[4]}[n*{dims[4]}+{gate}+u];",
        ]
        for (a, b, c), coefficient in angular.items():
            x = f"p{mapping[0]}[n*{dims[0]}+{sx + a}+u*{dx}]"
            y = f"p{mapping[1]}[n*{dims[1]}+{sy + b}+u*{dy}]"
            parts += [f"{{ scalar v=scalar({coefficient:.17g})*adj[u*{dz}+{c}];"]
            if 0 in roles:
                parts.append(f"gx[{a}]+=v*{y}*gate;")
            if 1 in roles:
                parts.append(f"gy[{b}]+=v*{x}*gate;")
            if 4 in roles:
                parts.append(f"gg+=v*{x}*{y};")
            parts.append("}")
        if 0 in roles:
            parts.append(
                f"for(int a=0;a<{dx};++a) atomicAdd(o0+n*{dims[0]}+{sx}+u*{dx}+a,gx[a]);"
            )
        if 1 in roles:
            parts.append(
                f"for(int b=0;b<{dy};++b) atomicAdd(o1+n*{dims[1]}+{sy}+u*{dy}+b,gy[b]);"
            )
        if 4 in roles:
            parts.append(f"atomicAdd(o4+n*{dims[4]}+{gate}+u,gg);")
        parts.append("} return; }")
        cases.append("\n".join(parts))
    count = max(i for _, mapping in branches for i in mapping) + 1
    pointers = ", ".join(f"const scalar* p{i}" for i in range(count))
    destinations = ", ".join(f"scalar* o{role}" for role in roles)
    scalar = "double" if dtype == torch.float64 else "float"
    return (
        f"""using scalar={scalar};
    extern "C" __global__ void run({pointers},const long long* types,{destinations},long long nodes) {{
       __shared__ scalar adj[{shared_size}];
       long long n=blockIdx.x; int t=threadIdx.x;
       switch(blockIdx.y) {{ {"".join(cases)} }}
    }}""",
        len(paths),
        count,
    )


@lru_cache(maxsize=128)
def weights_source(dtype, metadata, terms):
    plans = decode_metadata(metadata)
    paths = defaultdict(list)
    for term_index, (plan, mapping, _, slot) in enumerate(terms):
        for _, entries in plans[plan][2][3]:
            for r, _ in entries:
                paths[r[10]].append((plan, mapping, r, slot, term_index))
    cases = []
    shared_size = 1
    for index, (gate, entries) in enumerate(sorted(paths.items())):
        angular, maps = {}, {}
        width = 0
        for plan, mapping, r, slot, term_index in entries:
            shift = r[9] - gate
            width = max(width, shift + r[4])
            dx, dy, dz = r[6:9]
            sx, sy = r[13:15]
            angular[r[16], r[17], r[18]] = r[12]
            maps[plan, mapping, r[2], r[15], shift, r[4], r[5], slot, term_index] = r[
                11
            ]
        dims = plans[entries[0][0]][0]
        mapping = entries[0][1]
        shared_size = max(shared_size, width * dz)
        parts = [
            f"case {index}: {{",
            f"for(int u=t;u<{width};u+=blockDim.x) {{ scalar v[{dz}]={{}};",
        ]
        for (a, b, c), coefficient in angular.items():
            parts.append(
                f"v[{c}]+=scalar({coefficient:.17g})*p{mapping[0]}[n*{dims[0]}+{sx + a}+u*{dx}]*p{mapping[1]}[n*{dims[1]}+{sy + b}+u*{dy}];"
            )
        parts += [
            f"for(int m=0;m<{dz};++m) angular[u*{dz}+m]=v[m]*p{mapping[4]}[n*{dims[4]}+{gate}+u]; }}",
            "__syncthreads();",
        ]
        for (plan, mapping, sw, so, shift, ci, co, slot, _), scale in maps.items():
            z = "0" if plan else "types[n]"
            parts += [
                f"for(int q=t;q<{ci * co};q+=blockDim.x) {{ int u=q/{co},v=q%{co}; scalar sum=0;",
                f"for(int m=0;m<{dz};++m) sum+=angular[({shift}+u)*{dz}+m]*p{mapping[3]}[n*{dims[3]}+{so}+v*{dz}+m];",
                f"atomicAdd(o{slot}+{z}*{plans[plan][0][2]}+{sw}+q,scalar({scale:.17g})*sum); }}",
            ]
        parts.append("return; }")
        cases.append("\n".join(parts))
    count = max(i for _, mapping, _, _ in terms for i in mapping) + 1
    slots = tuple(sorted({slot for _, _, _, slot in terms}))
    pointers = ", ".join(f"const scalar* p{i}" for i in range(count))
    destinations = ", ".join(f"scalar* o{slot}" for slot in slots)
    scalar = "double" if dtype == torch.float64 else "float"
    return (
        f"""using scalar={scalar};
    extern "C" __global__ void run({pointers}, const long long* types, {destinations}, long long nodes) {{
      __shared__ scalar angular[{shared_size}];
      long long n=blockIdx.x; int t=threadIdx.x;
      switch(blockIdx.y) {{ {"".join(cases)} }}
    }}""",
        len(paths),
        count,
        slots,
    )


def launch(metadata, program, node_type, operands, outputs):

    if operands[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Bilinear CUDA kernels support float32 and float64.")
    operands = [value.contiguous() for value in operands]
    node_type = node_type.contiguous()
    for output in outputs:
        output.zero_()
    nodes = node_type.numel()
    if not nodes:
        return
    grouped = defaultdict(list)
    weights = defaultdict(list)
    reverse = defaultdict(list)
    for term in program:
        plan, mapping, role, slot = term
        if role == 3:
            grouped[role, slot, mapping[0], mapping[1], mapping[4]].append(term)
        elif role in (0, 1, 4):
            reverse[mapping[0], mapping[1], mapping[3], mapping[4]].append(term)
        else:
            weights[mapping[0], mapping[1], mapping[4]].append(term)
    device, dtype = operands[0].device, operands[0].dtype
    jobs = []
    for terms in reverse.values():
        by_role = defaultdict(list)
        for plan, mapping, role, slot in terms:
            by_role[role].append((plan, mapping, slot))
        roles = tuple(sorted(by_role))
        branches = tuple((p, m) for p, m, _ in by_role[roles[0]])
        if all(
            tuple((p, m) for p, m, _ in by_role[r]) == branches
            and len({s for _, _, s in by_role[r]}) == 1
            for r in roles
        ):
            code, groups, count = inputs_source(dtype, metadata, branches, roles)
            if groups:
                args = [v.data_ptr() for v in operands[:count]] + [node_type.data_ptr()]
                args += [outputs[by_role[r][0][2]].data_ptr() for r in roles] + [nodes]
                jobs.append((code, args, nodes, groups, 128, 0))
        else:
            for term in terms:
                grouped[term[2], term[3]].append(term)
    for key, terms in grouped.items():
        role, slot = key[:2]
        if role == 3:
            code, groups, count = forward_source(dtype, metadata, tuple(terms))
            grid = (nodes + 3) // 4
        else:
            code, groups, width, count = transpose_source(dtype, metadata, tuple(terms))
            grid = (nodes * width + 127) // 128
        if groups:
            args = [v.data_ptr() for v in operands[:count]] + [
                node_type.data_ptr(),
                outputs[slot].data_ptr(),
                nodes,
            ]
            jobs.append((code, args, grid, groups, 128, 0))
    for terms in weights.values():
        code, groups, count, slots = weights_source(dtype, metadata, tuple(terms))
        if groups:
            args = [v.data_ptr() for v in operands[:count]] + [node_type.data_ptr()]
            args += [outputs[slot].data_ptr() for slot in slots] + [nodes]
            jobs.append((code, args, nodes, groups, 128, 0))
    compiled = kernels([job[0] for job in jobs], device)
    runtime().launch(
        [(compiled[code], *rest) for code, *rest in jobs],
        torch.cuda.current_stream(device).cuda_stream,
    )
