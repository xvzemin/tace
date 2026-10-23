"""Channel-wise local products without expanded edge intermediates."""

from collections import defaultdict
from functools import lru_cache

import torch

from ..utils.metadata import parse_metadata
from .cuda import kernels, runtime


def local_product(metadata, *features):
    """Contract packed edge features using sparse channel-wise paths."""
    unique, mapping = [], []
    for feature in features:
        index = next((i for i, value in enumerate(unique) if value is feature), None)
        if index is None:
            index = len(unique)
            unique.append(feature)
        mapping.append(index)
    operands = [x.flatten(1) for x in unique]
    mapping.append(len(operands))
    operands.append(features[0].new_empty((0, operands[0].shape[-1])))
    program = repr(((tuple(mapping), len(features), 0, 0),))
    return channel_product(metadata, program, operands)[0]


@lru_cache(maxsize=128)
def product_source(dtype, metadata, role, required, shared):
    width, dims, rows = parse_metadata(metadata)
    groups = defaultdict(list)
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
            f"out[n*{dims[role] * width}+{offset * width}+c]+=sum; break; }}"
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


@torch.library.custom_op(
    "eqx::local_channel_product", mutates_args=(), device_types="cuda"
)
def channel_product(
    metadata: str, program: str, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate sparse diagonal products and their recursive transposes."""
    outputs = channel_product_fake(metadata, program, operands)
    for output in outputs:
        output.zero_()
    nodes = operands[0].shape[0]
    if not nodes:
        return outputs
    width = parse_metadata(metadata)[0]
    jobs = []
    for mapping, role, slot, required in parse_metadata(program):
        shared = tuple(operands[k].size(0) == 1 and nodes != 1 for k in mapping)
        source, groups = product_source(
            operands[0].dtype, metadata, role, required, shared
        )
        if groups:
            args = [operands[k].data_ptr() for k in mapping]
            args += [stride for k in mapping for stride in operands[k].stride()]
            args += [outputs[slot].data_ptr(), nodes]
            blocks = (
                (nodes + 127) // 128 * ((width + 31) // 32)
                if shared[role]
                else (nodes * width + 127) // 128
            )
            jobs.append((source, args, blocks, groups, 128, 0))
    compiled = kernels([job[0] for job in jobs], operands[0].device)
    runtime().launch(
        [(compiled[source], *rest) for source, *rest in jobs],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs


@channel_product.register_fake
def channel_product_fake(metadata, program, operands):
    width, dims, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (max(slot for _, _, slot, _ in terms) + 1)
    for mapping, role, slot, _ in terms:
        if outputs[slot] is None:
            rows = operands[mapping[role]].size(0)
            outputs[slot] = operands[0].new_empty(
                (1 if rows == 1 else operands[0].shape[0], dims[role] * width)
            )
    return outputs


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, operands = inputs
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*operands)


def backward(ctx, grad_outputs):
    operands = list(ctx.saved_tensors)
    values = list(operands)
    cotangents = {}
    for slot, grad in enumerate(grad_outputs):
        if grad is not None:
            cotangents[slot] = len(values)
            values.append(grad)
    terms, destinations = [], {}
    for mapping, output, slot, required in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and ctx.needs_input_grad[2][index]:
                destination = destinations.setdefault(index, len(destinations))
                # A transpose must not reintroduce monomials removed by an
                # earlier derivative with respect to an absent operand.
                terms.append(
                    (tuple(replacement), role, destination, required | (1 << output))
                )
    gradients = [None] * len(operands)
    if terms:
        results = channel_product(ctx.kernel_metadata, repr(tuple(terms)), values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, gradients


channel_product.register_autograd(backward, setup_context=setup_context)
