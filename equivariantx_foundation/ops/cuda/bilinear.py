"""Block-local bilinear products and factored transposed contractions."""

from collections import defaultdict
from functools import lru_cache
from itertools import chain

import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import decode_plans, encode_plans, parse_metadata
from equivariantx.utils.outputs import indexed_plan_outputs

from .. import bilinear


def evaluate(module, features, base, gates, weight, node_type, shared_weight=None):
    """Contract expert and shared contributions directly into the output."""
    program, operands = bilinear.bilinear_inputs(
        features, base, gates, weight, module.contraction_dim, shared_weight
    )
    output = bilinear.contract(module.kernel_metadata, program, node_type, operands)[0]
    return module.project_output(output)


@lru_cache(maxsize=128)
def metadata_batches(metadata, max_entries=1024):
    """Partition complete angular paths into bounded compilation schedules."""
    plans = decode_plans(metadata)
    sizes = defaultdict(int)
    for _, _, schedules in plans:
        for _, entries in schedules[3]:
            for record, _ in entries:
                sizes[record[10]] += 1
    batches, batch, size = [], [], 0
    for path, count in sorted(sizes.items()):
        if batch and size + count > max_entries:
            batches.append(set(batch))
            batch, size = [], 0
        batch.append(path)
        size += count
    if batch:
        batches.append(set(batch))
    if len(batches) <= 1:
        return (metadata,)
    result = []
    for paths in batches:
        selected = []
        for dims, weighted, schedules in plans:
            schedules = tuple(
                tuple(
                    (key, values)
                    for key, entries in groups
                    if (
                        values := tuple(
                            entry for entry in entries if entry[0][10] in paths
                        )
                    )
                )
                for groups in schedules
            )
            selected.append(repr((dims, weighted, schedules)))
        result.append(encode_plans(selected))
    return tuple(result)


@lru_cache(maxsize=128)
def forward_source(dtype, metadata, terms, extra_inputs=()):
    plans = decode_plans(metadata)
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
        factors = ((mapping[0], mapping[1], mapping[4]), *extra_inputs)
        if extra_inputs:
            parts.append("scalar total=0;")
        for i, (left, right, gate_input) in enumerate(factors):
            if i:
                parts.append("v=0;")
            for (sx, sy, dx, dy), coefficient in angular.items():
                parts.append(
                    f"v += scalar({coefficient:.17g})*p{left}[n*{dim[0]}+{sx}+u*{dx}]*p{right}[n*{dim[1]}+{sy}+u*{dy}];"
                )
            if extra_inputs:
                parts.append(f"total+=v*p{gate_input}[n*{dim[4]}+{gate}+u];")
        reduce = (
            "v=total;" if extra_inputs else f"v*=p{mapping[4]}[n*{dim[4]}+{gate}+u];"
        )
        parts += [
            f"{reduce} }} a[q]=v; }}",
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
    if extra_inputs:
        count = max(count, 1 + max(i for factors in extra_inputs for i in factors))
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
    plans = decode_plans(metadata)
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
def inputs_source(dtype, metadata, branches, roles, large_batch=False):
    plans = decode_plans(metadata)
    paths = defaultdict(list)
    for branch, (plan, mapping) in enumerate(branches):
        for _, entries in plans[plan][2][3]:
            for r, _ in entries:
                paths[r[10]].append((plan, mapping, r, branch))
    narrow_output = (
        large_batch
        and dtype == torch.float32
        and roles == (0, 1, 4)
        and len(branches) == 1
        and all(
            r[4:6] == (256, 64) for entries in paths.values() for _, _, r, _ in entries
        )
    )
    lanes = 8 if narrow_output else 32
    shuffle = (
        "__shfl_down_sync(0xffu << (t & 24),sum[m],s,8)"
        if narrow_output
        else "__shfl_down_sync(0xffffffff,sum[m],s)"
    )
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
        itemsize = 8 if dtype == torch.float64 else 4
        tile_width = min(width, max(1, 32 * ((32 << 10) // (itemsize * dz * 32))))
        shared_size = max(shared_size, tile_width * dz)
        parts = [
            f"case {index}: {{",
            f"for(int start=0;start<{width};start+={tile_width}) {{",
            f"for(int u=start+t/{lanes};u<{width} && u<start+{tile_width};u+=blockDim.x/{lanes}) {{ scalar sum[{dz}]={{}};",
        ]
        for (plan, indices, sw, so, shift, ci, co, _), scale in maps.items():
            z = "0" if plan else "types[n]"
            parts += [
                f"if(u>={shift} && u<{shift + ci}) {{ scalar part[{dz}]={{}};",
                f"for(int v=t%{lanes};v<{co};v+={lanes}) {{",
                f"scalar w=p{indices[2]}[{z}*{plans[plan][0][2]}+{sw}+(u-{shift})*{co}+v];",
                f"#pragma unroll\nfor(int m=0;m<{dz};++m) "
                f"part[m]+=w*p{indices[3]}[n*{dims[3]}+{so}+v*{dz}+m]; }}",
                f"#pragma unroll\nfor(int m=0;m<{dz};++m) "
                f"sum[m]+=scalar({scale:.17g})*part[m]; }}",
            ]
        parts += [
            f"#pragma unroll\nfor(int m=0;m<{dz};++m) {{",
            f"for(int s={lanes // 2};s>0;s/=2) sum[m]+={shuffle};",
            f"if(t%{lanes}==0) adj[(u-start)*{dz}+m]=sum[m]; }} }}",
            "__syncthreads();",
            f"for(int u=start+t;u<{width} && u<start+{tile_width};u+=blockDim.x) {{",
            f"scalar gx[{dx}]={{}}, gy[{dy}]={{}}, gg=0;",
            f"scalar gate=p{mapping[4]}[n*{dims[4]}+{gate}+u];",
        ]
        for (a, b, c), coefficient in angular.items():
            x = f"p{mapping[0]}[n*{dims[0]}+{sx + a}+u*{dx}]"
            y = f"p{mapping[1]}[n*{dims[1]}+{sy + b}+u*{dy}]"
            parts += [
                f"{{ scalar v=scalar({coefficient:.17g})*adj[(u-start)*{dz}+{c}];"
            ]
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
        parts.append("} __syncthreads(); } return; }")
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
        256 if narrow_output else 128,
    )


@lru_cache(maxsize=128)
def weights_source(dtype, metadata, terms, extra_inputs=()):
    plans = decode_plans(metadata)
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
        itemsize = 8 if dtype == torch.float64 else 4
        tile_width = min(width, max(1, 32 * ((32 << 10) // (itemsize * dz * 32))))
        shared_size = max(shared_size, tile_width * dz)
        parts = [
            f"case {index}: {{",
            f"for(int start=0;start<{width};start+={tile_width}) {{",
            f"for(int u=start+t;u<{width} && u<start+{tile_width};u+=blockDim.x) {{ scalar v[{dz}]={{}};",
        ]
        factors = ((mapping[0], mapping[1], mapping[4]), *extra_inputs)
        if extra_inputs:
            parts.append(f"scalar total[{dz}]={{}};")
        for i, (left, right, gate_input) in enumerate(factors):
            if i:
                parts.append(f"for(int m=0;m<{dz};++m) v[m]=0;")
            for (a, b, c), coefficient in angular.items():
                parts.append(
                    f"v[{c}]+=scalar({coefficient:.17g})*p{left}[n*{dims[0]}+{sx + a}+u*{dx}]*p{right}[n*{dims[1]}+{sy + b}+u*{dy}];"
                )
            if extra_inputs:
                parts.append(
                    f"for(int m=0;m<{dz};++m) total[m]+=v[m]*p{gate_input}[n*{dims[4]}+{gate}+u];"
                )
        result = (
            "total[m]" if extra_inputs else f"v[m]*p{mapping[4]}[n*{dims[4]}+{gate}+u]"
        )
        parts += [
            f"for(int m=0;m<{dz};++m) angular[(u-start)*{dz}+m]={result}; }}",
            "__syncthreads();",
        ]
        for (plan, mapping, sw, so, shift, ci, co, slot, _), scale in maps.items():
            z = "0" if plan else "types[n]"
            parts += [
                f"for(int q=t;q<{ci * co};q+=blockDim.x) {{ int u=q/{co},v=q%{co}; scalar sum=0;",
                f"if(u+{shift}<start || u+{shift}>=start+{tile_width}) continue;",
                f"for(int m=0;m<{dz};++m) sum+=angular[({shift}+u-start)*{dz}+m]*p{mapping[3]}[n*{dims[3]}+{so}+v*{dz}+m];",
                f"atomicAdd(o{slot}+{z}*{plans[plan][0][2]}+{sw}+q,scalar({scale:.17g})*sum); }}",
            ]
        parts.append("__syncthreads(); } return; }")
        cases.append("\n".join(parts))
    count = max(i for _, mapping, _, _ in terms for i in mapping) + 1
    if extra_inputs:
        count = max(count, 1 + max(i for factors in extra_inputs for i in factors))
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


@lru_cache(maxsize=128)
def factor_groups(groups, input_role):
    """Share the coefficient contraction across matching product-rule terms.

    Everything except the two features and their gate must match, including
    coefficient plan (element/shared), other factor, slot and term multiplicity.
    """
    by_output = defaultdict(list)
    for inputs, terms in groups:
        by_output[tuple((p, m[input_role], slot) for p, m, _, slot in terms)].append(
            (inputs, terms)
        )
    jobs = []
    for matched in by_output.values():
        if 2 <= len(matched) <= 3:
            jobs.append((matched[0][1], tuple(inputs for inputs, _ in matched[1:])))
        else:
            jobs.extend((terms, ()) for _, terms in matched)
    return tuple(jobs)


@lru_cache(maxsize=128)
def wide_projections(metadata):
    # For narrow coefficient matrices there is less work to amortize the
    # larger angular expression. Leave those schedules and mixed widths alone.
    return all(
        min(r[4:6]) >= 32
        for _, _, schedules in decode_plans(metadata)
        for _, entries in schedules[3]
        for r, _ in entries
    )


def execution_jobs(metadata, program, node_type, operands, outputs):
    """Build forward and transposed launches for one group of angular paths."""
    nodes = node_type.numel()
    grouped = defaultdict(list)
    forwards = defaultdict(list)
    weights = defaultdict(list)
    reverse = defaultdict(list)
    for term in program:
        plan, mapping, role, slot = term
        if role == 3:
            forwards[slot, (mapping[0], mapping[1], mapping[4])].append(term)
        elif role in (0, 1, 4):
            reverse[mapping[0], mapping[1], mapping[3], mapping[4]].append(term)
        else:
            weights[mapping[0], mapping[1], mapping[4]].append(term)
    dtype = operands[0].dtype
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
            code, groups, count, threads = inputs_source(
                dtype, metadata, branches, roles, nodes >= 1024
            )
            if groups:
                args = [v.data_ptr() for v in operands[:count]] + [node_type.data_ptr()]
                args += [outputs[by_role[r][0][2]].data_ptr() for r in roles] + [nodes]
                jobs.append((code, args, nodes, groups, threads, 0))
        else:
            for term in terms:
                grouped[term[2], term[3]].append(term)
    if len(forwards) > 1 and wide_projections(metadata):
        forward_jobs = factor_groups(
            tuple((k[1], tuple(v)) for k, v in forwards.items()), 2
        )
    else:
        forward_jobs = ((tuple(v), ()) for v in forwards.values())
    for terms, extra_inputs in chain(
        forward_jobs, ((tuple(v), ()) for v in grouped.values())
    ):
        _, _, role, slot = terms[0]
        if role == 3:
            code, groups, count = forward_source(dtype, metadata, terms, extra_inputs)
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
    if len(weights) > 1 and wide_projections(metadata):
        weight_jobs = factor_groups(tuple((k, tuple(v)) for k, v in weights.items()), 3)
    else:
        weight_jobs = ((tuple(v), ()) for v in weights.values())
    for terms, extra_inputs in weight_jobs:
        code, groups, count, slots = weights_source(
            dtype, metadata, terms, extra_inputs
        )
        if groups:
            args = [v.data_ptr() for v in operands[:count]] + [node_type.data_ptr()]
            args += [outputs[slot].data_ptr() for slot in slots] + [nodes]
            jobs.append((code, args, nodes, groups, 128, 0))
    return jobs


def launch(metadata, program, node_type, operands, outputs):
    """Compile and execute bounded angular schedules on the current stream."""
    if operands[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Bilinear CUDA kernels support float32 and float64.")
    operands = [value.contiguous() for value in operands]
    node_type = node_type.contiguous()
    for output in outputs:
        output.zero_()
    if not node_type.numel():
        return
    jobs = [
        job
        for batch in metadata_batches(metadata)
        for job in execution_jobs(batch, program, node_type, operands, outputs)
    ]
    device = operands[0].device
    compiled = kernels([job[0] for job in jobs], device)
    runtime().launch(
        [(compiled[code], *rest) for code, *rest in jobs],
        torch.cuda.current_stream(device).cuda_stream,
    )


def contract(
    metadata: str, program: str, node_type: torch.Tensor, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    from equivariantx.ops.common.cuda.graph import execute

    outputs = indexed_plan_outputs(metadata, program, node_type, operands)
    terms = parse_metadata(program)

    def run(inputs, outputs):
        launch(metadata, terms, inputs[0], inputs[1:], outputs)

    read = {0}
    for _, mapping, output, _ in terms:
        read.update(1 + index for role, index in enumerate(mapping) if role != output)
    execute("ace", (metadata, program), run, (node_type, *operands), outputs, read=read)
    return outputs
