"""CUDA evaluation of channel-wise products."""

from collections import defaultdict
from functools import lru_cache

import torch

from ....utils.metadata import (
    channel_product_layout,
    independent_channel_groups,
    parse_metadata,
)
from ....utils.outputs import channel_product_outputs
from .runtime import kernels, runtime


@lru_cache(maxsize=128)
def joint_product_source(metadata, program, dtype):
    """Fuse all live terms/slots only when the *actual* program has one writer
    per output coordinate and no holes (including aliased operand mappings).
    """
    width, dims, _ = parse_metadata(metadata)
    groups = independent_channel_groups(metadata)
    terms = parse_metadata(program)
    if not groups or len(terms) > 12 or dtype not in (torch.float64, torch.float32):
        return None
    owners, sizes, cases, reads = {}, {}, [], set()
    for group, rows in enumerate(groups):
        expressions = defaultdict(list)
        for mapping, role, slot, required in terms:
            if sizes.setdefault(slot, dims[role]) != dims[role]:
                return None
            subtotals = defaultdict(list)
            for indices, coefficient in rows:
                offset = indices[role]
                if offset < 0 or any(
                    i < 0 and required & (1 << k) for k, i in enumerate(indices)
                ):
                    continue
                factors = []
                for k, i in enumerate(indices):
                    if k != role and i >= 0:
                        p = mapping[k]
                        reads.add(p)
                        factors.append(f"p{p}[n*s{p}+({i * width}+c)*t{p}]")
                product = "*".join(factors) or "scalar(1)"
                subtotals[offset].append(f"v+=scalar({coefficient:.17g})*({product});")
            for offset, subtotal in subtotals.items():
                key = slot, offset
                if owners.setdefault(key, group) != group:
                    return None
                expressions[key].append("".join(subtotal))
        writes = []
        for (slot, i), subtotals in expressions.items():
            adds = "".join(
                f"{{scalar v=0;{subtotal} sum+=v;}}" for subtotal in subtotals
            )
            writes.append(
                f"{{scalar sum=0;{adds} q{slot}[n*{width * sizes[slot]}+{i * width}+c]=sum;}}"
            )
        cases.append(f"case {group}: {{ {''.join(writes)} break; }}")
    if len(owners) != sum(sizes.values()):
        return None
    # Only pass pointers used by a surviving monomial. Program mappings may
    # also contain empty shape markers or gradients unused at this AD order.
    inputs = sorted(reads)
    args = [f"const scalar* p{p},long long s{p},long long t{p}" for p in inputs]
    args += [f"scalar* q{k}" for k in range(len(sizes))]
    scalar = "double" if dtype == torch.float64 else "float"
    tiles = 128 // width
    source = f"""
        using scalar={scalar};
        extern "C" __global__ void run({",".join(args)}) {{
            long long n=blockIdx.x; int c=threadIdx.x%{width};
            switch(blockIdx.y*{tiles}+threadIdx.x/{width}) {{ {"".join(cases)} }}
        }}
    """
    return source, inputs, (len(groups) + tiles - 1) // tiles


@lru_cache(maxsize=128)
def strided_product_source(metadata, dtype, role, required, shared, accumulate):
    """Generate contractions for mixed multiplicities and Cartesian layouts."""
    dims, rows = channel_product_layout(metadata)
    groups = defaultdict(list)
    for indices, coefficient, width, strides in rows:
        if (
            width
            and indices[role] >= 0
            and all(
                index >= 0 for k, index in enumerate(indices) if required & (1 << k)
            )
        ):
            groups[indices[role], strides[role], width].append(
                (indices, coefficient, strides)
            )
    cases = []
    counts = defaultdict(int)
    for offset, stride, width in groups:
        for index in {offset + c * stride for c in range(width)}:
            counts[index] += 1
    for group, ((offset, stride, width), entries) in enumerate(groups.items()):
        terms = []
        for indices, coefficient, steps in entries:
            factors = (
                "*".join(
                    f"p{k}[{'0' if shared[k] else f'n*s{k}'}+({index}+c*{steps[k]})*t{k}]"
                    for k, index in enumerate(indices)
                    if k != role and index >= 0
                )
                or "scalar(1)"
            )
            terms.append(f"sum+=scalar({coefficient:.17g})*({factors});")
        if shared[role]:
            body = f"""
                const int tiles=({width}+31)/32;
                int c=(blockIdx.x%tiles)*32+threadIdx.x%32;
                long long begin=(blockIdx.x/tiles)*128;
                scalar sum=0;
                if(c<{width}) {{
                    for(long long n=begin+threadIdx.x/32;n<nodes && n<begin+128;n+=4) {{
                        {"".join(terms)}
                    }}
                }}
                partial[threadIdx.x]=sum;
                __syncthreads();
                if(threadIdx.x<32 && c<{width}) {{
                    sum+=partial[threadIdx.x+32]+partial[threadIdx.x+64]+partial[threadIdx.x+96];
                    atomicAdd(out+{offset}+c*{stride},sum);
                }}
            """
        else:
            # Broadcast channels can overlap paths of different widths.
            atomic = not stride or any(
                counts[offset + c * stride] > 1 for c in range(width)
            )
            store = (
                f"atomicAdd(out+n*{dims[role]}+{offset}+c*{stride},sum);"
                if atomic
                else f"out[n*{dims[role]}+{offset}+c*{stride}]{'+=' if accumulate else '='}sum;"
            )
            body = f"""
                long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
                if(i>=nodes*{width}) return;
                long long n=i/{width}; int c=i%{width}; scalar sum=0;
                {"".join(terms)}
                {store}
            """
        cases.append(f"case {group}: {{ {body} break; }}")
    pointers = ",".join(f"const scalar* __restrict__ p{k}" for k in range(len(dims)))
    strides = ",".join(f"long long s{k}, long long t{k}" for k in range(len(dims)))
    source = f"""
        using scalar={"double" if dtype == torch.float64 else "float"};
        extern "C" __global__ void run({pointers}, {strides}, scalar* out, long long nodes) {{
            {"__shared__ scalar partial[128];" if shared[role] else ""}
            switch(blockIdx.y) {{ {"".join(cases)} }}
        }}
    """
    full_write = (
        not shared[role]
        and len(counts) == dims[role]
        and all(n == 1 for n in counts.values())
        and all(stride != 0 for _, stride, _ in groups)
    )
    return source, len(groups), max((key[2] for key in groups), default=1), full_write


@lru_cache(maxsize=128)
def product_source(dtype, metadata, role, required, shared, accumulate):
    width, dims, rows = parse_metadata(metadata)
    groups = defaultdict(list)
    if not shared[role] and not accumulate:
        for offset in range(dims[role]):
            groups[offset] = []
    for indices, coefficient in rows:
        if indices[role] >= 0 and all(
            index >= 0 for k, index in enumerate(indices) if required & (1 << k)
        ):
            groups[indices[role]].append((indices, coefficient))
    cases = []
    for group, (offset, entries) in enumerate(sorted(groups.items())):
        terms = []
        for indices, coefficient in entries:
            factors = (
                "*".join(
                    f"p{k}[{'0' if shared[k] else f'n*s{k}'}+({index * width}+c)*t{k}]"
                    for k, index in enumerate(indices)
                    if k != role and index >= 0
                )
                or "scalar(1)"
            )
            terms.append(f"sum+=scalar({coefficient:.17g})*({factors});")
        cases.append(
            f"case {group}: {{ {''.join(terms)} break; }}"
            if shared[role]
            else f"case {group}: {{ scalar sum=0; {''.join(terms)} "
            f"out[n*{dims[role] * width}+{offset * width}+c]{'+=' if accumulate else '='}sum; break; }}"
        )
    pointers = ",".join(f"const scalar* p{k}" for k in range(len(dims)))
    strides = ",".join(f"long long s{k}, long long t{k}" for k in range(len(dims)))
    scalar = "double" if dtype == torch.float64 else "float"
    offsets = ",".join(str(offset * width) for offset in sorted(groups)) or "0"
    if shared[role]:
        # Reduce a bounded row tile before writing shared-operand gradients.
        body = f"""
        const int tiles=({width}+31)/32;
        int c=(blockIdx.x%tiles)*32+threadIdx.x%32;
        long long begin=(blockIdx.x/tiles)*128;
        scalar sum=0;
        if(c<{width}) {{
            for(long long n=begin+threadIdx.x/32;n<nodes && n<begin+128;n+=4) {{
                switch(blockIdx.y) {{ {"".join(cases)} }}
            }}
        }}
        __shared__ scalar partial[128];
        partial[threadIdx.x]=sum;
        __syncthreads();
        if(threadIdx.x<32 && c<{width}) {{
            sum+=partial[threadIdx.x+32]+partial[threadIdx.x+64]+partial[threadIdx.x+96];
            atomicAdd(out+offsets[blockIdx.y]+c,sum);
        }}
        """
    else:
        body = f"""
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=nodes*{width}) return;
        long long n=i/{width}; int c=i%{width};
        switch(blockIdx.y) {{ {"".join(cases)} }}
        """
    constants = (
        f"__device__ __constant__ int offsets[]={{{offsets}}};" if shared[role] else ""
    )
    return (
        f"""
    using scalar={scalar};
    {constants}
    extern "C" __global__ void run({pointers}, {strides}, scalar* out, long long nodes) {{
        {body}
    }}""",
        len(groups),
    )


def channel_product(
    metadata: str, program: str, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate sparse diagonal products and their recursive transposes."""
    outputs = channel_product_outputs(metadata, program, operands)
    nodes = operands[0].shape[0]
    if not nodes:
        for output in outputs:
            output.zero_()
        return outputs
    width, dims, _ = parse_metadata(metadata)
    if width and len(dims) == 4 and independent_channel_groups(metadata) is not None:
        joint = joint_product_source(metadata, program, operands[0].dtype)
        if joint:
            source, inputs, grid_y = joint
            if any(operands[p].size(0) == 0 for p in inputs):
                raise ValueError(
                    "Cannot read a zero-row operand in a nonempty channel product."
                )
        # Shared inputs/outputs require the general reduction. Empty shape
        # markers are safe only when the generated kernel doesn't read them.
        if (
            joint
            and all(x.size(0) == nodes for x in outputs)
            and all(operands[p].size(0) == nodes for p in inputs)
        ):
            args = []
            for p in inputs:
                args.extend((operands[p].data_ptr(), *operands[p].stride()))
            args.extend(x.data_ptr() for x in outputs)
            device = operands[0].device
            kernel = kernels([source], device)[source]
            runtime().launch(
                [(kernel, args, nodes, grid_y, 128, 0)],
                torch.cuda.current_stream(device).cuda_stream,
            )
            return outputs
    jobs = []
    initialized = set()
    for mapping, role, slot, required in parse_metadata(program):
        shared = tuple(operands[k].size(0) == 1 and nodes != 1 for k in mapping)
        accumulate = slot in initialized
        if width:
            source, groups = product_source(
                operands[0].dtype, metadata, role, required, shared, accumulate
            )
            channels = width
            full_write = not shared[role]
        else:
            source, groups, channels, full_write = strided_product_source(
                metadata, operands[0].dtype, role, required, shared, accumulate
            )
        if not full_write and not accumulate:
            outputs[slot].zero_()
        initialized.add(slot)
        if groups:
            args = [operands[k].data_ptr() for k in mapping]
            args += [stride for k in mapping for stride in operands[k].stride()]
            args += [outputs[slot].data_ptr(), nodes]
            blocks = (
                (nodes + 127) // 128 * ((channels + 31) // 32)
                if shared[role]
                else (nodes * channels + 127) // 128
            )
            jobs.append((source, args, blocks, groups, 128, 0))
    compiled = kernels([job[0] for job in jobs], operands[0].device)
    runtime().launch(
        [(compiled[source], *rest) for source, *rest in jobs],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs
