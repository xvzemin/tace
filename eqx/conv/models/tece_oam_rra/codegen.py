"""Cooperative CUDA evaluation of fixed-width local interaction expressions."""

from collections import Counter, defaultdict
from functools import lru_cache

import torch

from .program import decode


@lru_cache(maxsize=128)
def source(
    dtype, metadata, mode="edge", heads=0, cutoff=4, eps=0.0, channels=0, splits=1
):
    """Generate a fused edge program or receiver-wise online normalization.

    Large parameter arrays and their outer-product adjoints are read or written
    directly. Only local feature vectors occupy the reusable block workspace.
    """
    nodes, outputs = decode(metadata)
    roots = tuple(x[0] for x in outputs)
    live = set()

    def visit(i):
        if i not in live:
            live.add(i)
            for arg in nodes[i][2]:
                visit(arg)

    for i in roots:
        visit(i)
    uses = Counter(arg for i in live for arg in nodes[i][2])
    dependent = []
    for op, _, args, data in nodes:
        dependent.append(
            data[1] != "shared" if op == "input" else any(dependent[j] for j in args)
        )
    stored = {
        i
        for i in live
        if nodes[i][1]
        and (
            nodes[i][0] in ("scatter", "product")
            or (nodes[i][0] == "matmul" and (nodes[i][3][1] > 2 or nodes[i][1] <= 256))
            or (
                uses[i] > 1
                and nodes[i][0]
                in (
                    "add",
                    "mul",
                    "reciprocal",
                    "silu",
                    "sigmoid",
                    "tanh",
                    "exp",
                    "sin",
                    "cos",
                )
            )
        )
        and nodes[i][1] <= 16384
        # Scaled/transposed parameter matrices can be read directly. Copying
        # them into each edge's workspace dominates wide derivative programs.
        and (dependent[i] or nodes[i][1] <= 256)
    }
    order = sorted(stored)
    position = {i: k for k, i in enumerate(order)}

    @lru_cache(None)
    def buffers(i):
        if i in stored:
            return frozenset((i,))
        return frozenset().union(*(buffers(j) for j in nodes[i][2]))

    last = dict(position)
    for event, i in enumerate(order):
        for arg in nodes[i][2]:
            for j in buffers(arg):
                last[j] = max(last[j], event)
    for i in roots:
        for j in buffers(i):
            last[j] = len(order)
    offsets, free, pending, capacity = {}, [], [], 0
    for event, i in enumerate(order):
        for j in tuple(pending):
            if last[j] < event:
                free.append((offsets[j], nodes[j][1]))
                pending.remove(j)
        # Coalesce adjacent released vectors before allocating another one.
        merged = []
        for start, size in sorted(free):
            if merged and sum(merged[-1]) == start:
                old, length = merged.pop()
                merged.append((old, length + size))
            else:
                merged.append((start, size))
        free = merged
        size = nodes[i][1]
        match = next((k for k, (_, length) in enumerate(free) if length >= size), None)
        if match is None:
            offsets[i] = capacity
            capacity += size
        else:
            start, length = free.pop(match)
            offsets[i] = start
            if length > size:
                free.append((start + size, length - size))
        pending.append(i)

    declarations, tables = [], {}

    def table(values):
        values = tuple(values)
        if values not in tables:
            name = f"indices{len(tables)}"
            tables[values] = name
            declarations.append(
                f"__device__ const int {name}[]={{{','.join(map(str, values)) or '0'}}};"
            )
        return tables[values]

    def get(i, index):
        return f"v{i}({index})"

    def expression(i):
        op, size, args, data = nodes[i]
        a = get(args[0], "i") if args else ""
        if op == "input":
            slot, kind = data
            offset = {
                "shared": "0",
                "edge": "edge",
                "source": "source",
                "target": "target",
            }[kind]
            return f"return p{slot}[{offset}*{size}+i];"
        if op == "constant":
            return f"return scalar({data:.17g});"
        if op in ("add", "mul"):
            return f"return {a} {'+' if op == 'add' else '*'} {get(args[1], 'i')};"
        if op == "reciprocal":
            return f"return scalar(1)/{a};"
        if op in ("exp", "sin", "cos", "tanh"):
            return f"return {op}({a});"
        if op in ("sigmoid", "silu"):
            # Match the framework's pointwise functions, including their finite
            # precision saturation, rather than clamping the activation input.
            return f"scalar x={a}; return {'x' if op == 'silu' else 'scalar(1)'}/(scalar(1)+exp(-x));"
        if op == "slice":
            return f"return {get(args[0], f'i+{data}')};"
        if op == "unslice":
            return f"return i>={data} && i<{data + nodes[args[0]][1]} ? {get(args[0], f'i-{data}')} : scalar(0);"
        if op == "concat":
            terms, offset = [], 0
            for arg in args:
                offset += nodes[arg][1]
                terms.append(
                    f"if(i<{offset}) return {get(arg, f'i-{offset - nodes[arg][1]}')};"
                )
            return " ".join(terms) + " return scalar(0);"
        if op == "transpose":
            rows, columns = data
            return f"return {get(args[0], f'(i%{rows})*{columns}+i/{rows}')};"
        if op == "gather":
            return f"return {get(args[0], f'{table(data)}[i]')};"
        if op == "scatter":
            groups = defaultdict(list)
            for j, k in enumerate(data):
                groups[k].append(j)
            indices, ptr = [], [0]
            for j in range(size):
                indices.extend(groups[j])
                ptr.append(len(indices))
            return f"scalar sum=0; for(int k={table(ptr)}[i];k<{table(ptr)}[i+1];++k) sum+={get(args[0], f'{table(indices)}[k]')}; return sum;"
        if op == "matmul":
            _, inner, columns = data
            return f"scalar sum=0; for(int k=0;k<{inner};++k) sum+={get(args[0], f'(i/{columns})*{inner}+k')}*{get(args[1], f'k*{columns}+i%{columns}')}; return sum;"
        if op == "product":
            (width, dims, entries), role, required = data
            groups = defaultdict(list)
            for indices, coefficient in entries:
                if indices[role] >= 0 and all(
                    index >= 0 for k, index in enumerate(indices) if required & (1 << k)
                ):
                    groups[indices[role]].append((indices, coefficient))
            cases = []
            for group, terms in sorted(groups.items()):
                factors = []
                for indices, coefficient in terms:
                    factors.append(
                        f"sum+=scalar({coefficient:.17g})"
                        + "".join(
                            f"*{get(args[k], f'{index * width}+c')}"
                            for k, index in enumerate(indices)
                            if k != role and index >= 0
                        )
                        + ";"
                    )
                cases.append(f"case {group}: {{ {''.join(factors)} break; }}")
            return f"int c=i%{width}; scalar sum=0; switch(i/{width}) {{ {''.join(cases)} }} return sum;"
        raise NotImplementedError(op)

    slots = 1 + max((node[3][0] for node in nodes if node[0] == "input"), default=-1)
    methods = []
    for i in sorted(live):
        body = expression(i)
        if i in stored:
            methods.append(
                f"__device__ __forceinline__ scalar r{i}(int i) const {{ {body} }}"
            )
            body = f"return buffer[{offsets[i]}+i];"
        methods.append(
            f"__device__ __forceinline__ scalar v{i}(int i) const {{ {body} }}"
        )
    stages = []
    for j in order:
        op, size, args, data = nodes[j]
        if (
            op == "matmul"
            and nodes[args[1]][0] == "transpose"
            and not dependent[args[1]]
            and data[1] >= 64
        ):
            # Adjoint channel maps read transposed weights. Reduce each dot
            # product across a warp so neighboring lanes read adjacent weights.
            _, inner, columns = data
            stages.append(f"""
            for(int i=threadIdx.x/32;i<{size};i+=blockDim.x/32) {{
              scalar sum=0;
              for(int k=threadIdx.x%32;k<{inner};k+=32)
                sum+=state.v{args[0]}((i/{columns})*{inner}+k)*state.v{args[1]}(k*{columns}+i%{columns});
              for(int stride=16;stride;stride/=2) sum+=__shfl_down_sync(0xffffffffu,sum,stride);
              if(threadIdx.x%32==0) state.buffer[{offsets[j]}+i]=sum;
            }}
            __syncthreads();""")
        else:
            stages.append(
                f"for(int i=threadIdx.x;i<{size};i+=blockDim.x) state.buffer[{offsets[j]}+i]=state.r{j}(i); __syncthreads();"
            )
    stages = "\n".join(stages)
    pointers = ", ".join(f"const scalar* p{i}" for i in range(slots))
    initializer = ",".join(f"p{i}" for i in range(slots))
    fields = " ".join(f"const scalar* p{i};" for i in range(slots))
    if mode == "online":
        score, value = roots
        width = nodes[value][1]
        output_offset = capacity
        capacity += width
        denominator = (
            "denominator[h]" if splits > 1 else f"(denominator[h]+scalar({eps:.17g}))"
        )
        normalized = (
            "accum[i]"
            if splits > 1
            else f"accum[i]/(denominator[h]+scalar({eps:.17g}))"
        )
        maximum = (
            "maximum[h]"
            if splits > 1
            else "(maximum[h]==-scalar(1.0/0.0)?scalar(0):maximum[h])"
        )
        body = f"""
        __shared__ scalar maximum[{heads}], denominator[{heads}], correction[{heads}], weight[{heads}];
        scalar* accum=buffer+{output_offset};
        for(long long task=blockIdx.x;task<count;task+=gridDim.x) {{
          long long node=task/{splits}, part=task%{splits};
          for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{ maximum[h]=-scalar(1.0/0.0); denominator[h]=0; }}
          for(int i=threadIdx.x;i<{width};i+=blockDim.x) accum[i]=0;
          __syncthreads();
          long long degree=ptr[node+1]-ptr[node];
          long long begin=ptr[node]+degree*part/{splits}, end=ptr[node]+degree*(part+1)/{splits};
          for(long long j=begin;j<end;++j) {{
            long long edge=order[j]; State state{{{initializer}, buffer, edge, source[edge], node}};
            {stages}
            for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{
              scalar score=state.v{score}(h), next=fmax(maximum[h],score);
              correction[h]=exp(maximum[h]-next);
              scalar c=p{cutoff}[edge], exponential=exp(score-next);
              denominator[h]=denominator[h]*correction[h]+exponential*c;
              weight[h]=exponential*c*c;
              maximum[h]=next;
            }}
            __syncthreads();
            for(int i=threadIdx.x;i<{width};i+=blockDim.x) {{
              int h=(i%{channels})/{channels // heads};
              accum[i]=correction[h]*accum[i]+weight[h]*state.v{value}(i);
            }}
            __syncthreads();
          }}
          for(int i=threadIdx.x;i<{width};i+=blockDim.x) {{
            int h=(i%{channels})/{channels // heads};
            out0[task*{width}+i]={normalized};
          }}
          for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{
            out1[task*{heads}+h]={denominator};
            out2[task*{heads}+h]={maximum};
          }}
          __syncthreads();
        }}"""
        count_outputs = 3
    else:
        destinations = sorted(set(slot for _, slot, _ in outputs))
        writes = []
        for root, slot, kind in outputs:
            size = nodes[root][1]
            offset = {
                "shared": "0",
                "source": "state.source",
                "target": "state.target",
                "edge": "edge",
            }[kind]
            index = f"{offset}*{size}+i"
            value = f"state.v{root}(i)"
            target = f"out{destinations.index(slot)}"
            store = (
                f"{target}[{index}]+=value;"
                if kind == "edge"
                else f"if(value!=scalar(0)) atomicAdd({target}+{index},value);"
            )
            writes.append(
                f"for(int i=threadIdx.x;i<{size};i+=blockDim.x) {{ scalar value={value}; {store} }} __syncthreads();"
            )
        body = f"""
        for(long long edge=blockIdx.x;edge<count;edge+=gridDim.x) {{
          State state{{{initializer}, buffer, edge, source[edge], target[edge]}};
          {stages}
          {"".join(writes)}
        }}"""
        count_outputs = len(destinations)
    output_args = ", ".join(f"scalar* out{i}" for i in range(count_outputs))
    scalar = "double" if dtype == torch.float64 else "float"
    code = f"""
    using scalar={scalar};
    {"".join(declarations)}
    struct State {{
      {fields} scalar* buffer; long long edge,source,target;
      {"".join(methods)}
    }};
    extern "C" __global__ void run({pointers},const long long* source,const long long* target,
       const long long* order,const long long* ptr,scalar* workspace,{output_args},long long count) {{
      extern __shared__ scalar local[];
      scalar* buffer=workspace?workspace+(long long)blockIdx.x*{capacity}:local;
      {body}
    }}"""
    return code, capacity, slots


def launch(
    metadata,
    inputs,
    source_index,
    target_index,
    outputs,
    *,
    mode="edge",
    heads=0,
    channels=0,
    eps=0.0,
):
    """Launch a native program with shared or bounded overflow workspace."""
    from ....kernels.cuda import kernels, runtime
    from ...graph import prepare_graph

    if any(x.device != inputs[0].device or x.dtype != inputs[0].dtype for x in inputs):
        raise ValueError("Local interaction inputs must share one device and dtype.")
    if source_index.dtype != torch.long or target_index.dtype != torch.long:
        raise TypeError("Local interaction edge indices must have dtype int64.")
    if (
        source_index.device != inputs[0].device
        or target_index.device != inputs[0].device
    ):
        raise ValueError("Edge indices and features must be on the same device.")
    online = mode == "online"
    count = inputs[0].shape[0] if online else source_index.numel()
    if not count:
        return
    processors = torch.cuda.get_device_properties(
        inputs[0].device
    ).multi_processor_count
    splits = 1
    if online:
        splits = min(
            8,
            max(1, (4 * processors + count - 1) // count),
            max(1, source_index.numel() // (16 * count)),
        )
    code, storage, slots = source(
        inputs[0].dtype, metadata, mode, heads, 4, eps, channels, splits
    )
    compiled = kernels([code], inputs[0].device)
    bytes_per_block = storage * inputs[0].element_size()
    shared = (
        bytes_per_block + (4 * heads * inputs[0].element_size() if online else 0)
        <= 49152
    )
    tasks = count * splits
    threads = 256 if storage >= 8192 else 128
    # Overflow storage remains bounded by the device, not the edge count.
    # Give every SM work instead of serializing wide programs onto 32 blocks.
    blocks = tasks if shared else min(tasks, 2 * processors)
    workspace = None if shared else inputs[0].new_empty((blocks, storage))
    order = ptr = None
    if online:
        order = prepare_graph(source_index, target_index, 1)
        counts = target_index.new_zeros(count).index_add(
            0, target_index, torch.ones_like(target_index)
        )
        ptr = torch.cat((target_index.new_zeros(1), counts.cumsum(0)))
    partials = outputs
    if splits > 1:
        partials = [
            value.new_empty((count, splits, *value.shape[1:])) for value in outputs
        ]
    arguments = [x.data_ptr() for x in inputs[:slots]]
    arguments += [
        x.data_ptr() if x is not None else 0
        for x in (source_index, target_index, order, ptr, workspace)
    ]
    arguments += [x.data_ptr() for x in partials] + [tasks]
    runtime().launch(
        [
            (
                compiled[code],
                arguments,
                blocks,
                1,
                threads,
                bytes_per_block if shared else 0,
            )
        ],
        torch.cuda.current_stream(inputs[0].device).cuda_stream,
    )
    if splits > 1:
        from ...attention import merge_attention

        merge_attention(*partials, channels, eps, outputs)
