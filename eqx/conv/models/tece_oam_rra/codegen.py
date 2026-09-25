"""Cooperative CUDA evaluation of fixed-width local interaction expressions."""

from collections import Counter, defaultdict
from functools import lru_cache

import torch

from .program import decode


@lru_cache(maxsize=128)
def source(dtype, metadata, mode="edge", heads=0, cutoff=4, eps=0.0):
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
    stages = "\n".join(
        f"for(int i=threadIdx.x;i<{nodes[j][1]};i+=blockDim.x) state.buffer[{offsets[j]}+i]=state.r{j}(i); __syncthreads();"
        for j in order
    )
    pointers = ", ".join(f"const scalar* p{i}" for i in range(slots))
    initializer = ",".join(f"p{i}" for i in range(slots))
    fields = " ".join(f"const scalar* p{i};" for i in range(slots))
    if mode == "stats":
        score = roots[0]
        body = f"""
        __shared__ scalar maximum[{heads}], denominator[{heads}];
        for(long long node=blockIdx.x;node<count;node+=gridDim.x) {{
          for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{ maximum[h]=-scalar(1.0/0.0); denominator[h]=0; }}
          __syncthreads();
          for(long long j=ptr[node];j<ptr[node+1];++j) {{
            long long edge=order[j]; State state{{{initializer}, buffer, edge, source[edge], node}};
            {stages}
            for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{
              scalar score=state.v{score}(h), next=fmax(maximum[h],score);
              denominator[h]=denominator[h]*exp(maximum[h]-next)+exp(score-next)*p{cutoff}[edge];
              maximum[h]=next;
            }}
            __syncthreads();
          }}
          for(int h=threadIdx.x;h<{heads};h+=blockDim.x) {{
            out0[node*{heads}+h]=denominator[h]+scalar({eps:.17g});
            out1[node*{heads}+h]=maximum[h]==-scalar(1.0/0.0)?scalar(0):maximum[h];
          }}
          __syncthreads();
        }}"""
        count_outputs = 2
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
    count = inputs[0].shape[0] if mode == "stats" else source_index.numel()
    if not count:
        return
    code, storage, slots = source(inputs[0].dtype, metadata, mode, heads, 4, eps)
    compiled = kernels([code], inputs[0].device)
    bytes_per_block = storage * inputs[0].element_size()
    shared = (
        bytes_per_block
        + (2 * heads * inputs[0].element_size() if mode == "stats" else 0)
        <= 49152
    )
    blocks = count if shared else min(count, 32)
    workspace = None if shared else inputs[0].new_empty((blocks, storage))
    order = ptr = None
    if mode == "stats":
        order = prepare_graph(source_index, target_index, 1)
        counts = target_index.new_zeros(count).index_add(
            0, target_index, torch.ones_like(target_index)
        )
        ptr = torch.cat((target_index.new_zeros(1), counts.cumsum(0)))
    arguments = [x.data_ptr() for x in inputs[:slots]]
    arguments += [
        x.data_ptr() if x is not None else 0
        for x in (source_index, target_index, order, ptr, workspace)
    ]
    arguments += [x.data_ptr() for x in outputs] + [count]
    runtime().launch(
        [(compiled[code], arguments, blocks, 1, 128, bytes_per_block if shared else 0)],
        torch.cuda.current_stream(inputs[0].device).cuda_stream,
    )
