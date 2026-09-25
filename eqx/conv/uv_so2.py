"""Channel-wise local products without expanded edge intermediates."""

from collections import defaultdict
from functools import lru_cache

import torch

from .ace.contraction import parse
from ..kernels.cuda import kernels, runtime


def local_product(metadata, *features):
    """Contract edge-local features into the layout of the first input."""
    operands = [x.flatten(1) for x in features]
    operands.append(features[0].new_empty((0, operands[0].shape[-1])))
    program = repr(((tuple(range(len(operands))), len(features), 0),))
    return channel_product(metadata, program, operands)[0]


@lru_cache(maxsize=128)
def product_source(dtype, metadata, role):
    width, dims, rows = parse(metadata)
    groups = defaultdict(list)
    for indices, coefficient in rows:
        groups[indices[role]].append((indices, coefficient))
    cases = []
    for group, (offset, entries) in enumerate(sorted(groups.items())):
        terms = []
        for indices, coefficient in entries:
            factors = "*".join(
                f"p{k}[n*s{k}+({index * width}+c)*t{k}]"
                for k, index in enumerate(indices)
                if k != role
            )
            terms.append(f"sum+=scalar({coefficient:.17g})*({factors});")
        cases.append(
            f"case {group}: {{ scalar sum=0; {''.join(terms)} "
            f"out[n*{dims[role] * width}+{offset * width}+c]+=sum; break; }}"
        )
    pointers = ",".join(f"const scalar* p{k}" for k in range(len(dims)))
    strides = ",".join(f"long long s{k}, long long t{k}" for k in range(len(dims)))
    scalar = "double" if dtype == torch.float64 else "float"
    return (
        f"""
    using scalar={scalar};
    extern "C" __global__ void run({pointers}, {strides}, scalar* out, long long nodes) {{
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=nodes*{width}) return;
        long long n=i/{width}; int c=i%{width};
        switch(blockIdx.y) {{ {"".join(cases)} }}
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
    width = parse(metadata)[0]
    jobs = []
    for mapping, role, slot in parse(program):
        source, groups = product_source(operands[0].dtype, metadata, role)
        if groups:
            args = [operands[k].data_ptr() for k in mapping]
            args += [stride for k in mapping for stride in operands[k].stride()]
            args += [outputs[slot].data_ptr(), nodes]
            jobs.append((source, args, (nodes * width + 127) // 128, groups, 128, 0))
    compiled = kernels([job[0] for job in jobs], operands[0].device)
    runtime().launch(
        [(compiled[source], *rest) for source, *rest in jobs],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs


@channel_product.register_fake
def channel_product_fake(metadata, program, operands):
    width, dims, _ = parse(metadata)
    terms = parse(program)
    outputs = [None] * (max(slot for _, _, slot in terms) + 1)
    for _, role, slot in terms:
        if outputs[slot] is None:
            outputs[slot] = operands[0].new_empty(
                (operands[0].shape[0], dims[role] * width)
            )
    return outputs


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, operands = inputs
    ctx.program = parse(program)
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
    for mapping, output, slot in ctx.program:
        if slot not in cotangents:
            continue
        replacement = list(mapping)
        replacement[output] = cotangents[slot]
        for role, index in enumerate(mapping):
            if role != output and ctx.needs_input_grad[2][index]:
                destination = destinations.setdefault(index, len(destinations))
                terms.append((tuple(replacement), role, destination))
    gradients = [None] * len(operands)
    if terms:
        results = channel_product(ctx.kernel_metadata, repr(tuple(terms)), values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, gradients


channel_product.register_autograd(backward, setup_context=setup_context)
