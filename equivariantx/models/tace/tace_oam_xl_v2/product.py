"""Node CG programs with one write per output, including mixed derivatives.

The ordinary product backend dispatches each derivative term separately. Here
all terms contributing to an output component share one accumulator. Inputs,
their strides and learned weights remain runtime values. Unsupported layouts
retain the ordinary EQX implementation.
"""

from collections import defaultdict
from functools import lru_cache

import torch

from equivariantx.ops.kernels.cuda.channel_product import channel_product as reference_product
from equivariantx.ops.kernels.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import channel_product_outputs


@lru_cache(maxsize=128)
def product_source(metadata, program, dtype, num_operands):
    """Generate a joint value/adjoint program in channel-contiguous layout."""
    width, dims, rows = parse_metadata(metadata)
    terms = parse_metadata(program)
    output_dims = {}
    groups = defaultdict(list)
    for mapping, role, slot, required in terms:
        if slot in output_dims and output_dims[slot] != dims[role]:
            raise ValueError("Product terms disagree on their output width")
        output_dims[slot] = dims[role]
        entries = defaultdict(list)
        for indices, coefficient in rows:
            if indices[role] < 0 or any(
                index < 0 and required & (1 << k) for k, index in enumerate(indices)
            ):
                continue
            factors = [
                f"p{mapping[k]}[n*s{mapping[k]}+({index * width}+c)*t{mapping[k]}]"
                for k, index in enumerate(indices)
                if k != role and index >= 0
            ]
            expression = "*".join(factors) or "scalar(1)"
            entries[indices[role]].append(
                f"part+=scalar({coefficient:.17g})*({expression});"
            )
        for index, expressions in entries.items():
            groups[slot, index].append(
                "{scalar part=0;" + "".join(expressions) + "total+=part;}"
            )
    cases = []
    for slot, dim in sorted(output_dims.items()):
        for index in range(dim):
            cases.append(
                f"case {len(cases)}: {{scalar total=0;"
                + "".join(groups[slot, index])
                + f"o{slot}[n*{dim * width}+{index * width}+c]=total;break;}}"
            )
    pointers = ",".join(f"const scalar* p{i}" for i in range(num_operands))
    strides = ",".join(f"long long s{i},long long t{i}" for i in range(num_operands))
    outputs = ",".join(f"scalar* o{i}" for i in range(len(output_dims)))
    scalar = "double" if dtype == torch.float64 else "float"
    return (
        f"""
    using scalar={scalar};
    extern "C" __global__ void run({pointers},{strides},{outputs},long long nodes) {{
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=nodes*{width}) return;
        long long n=i/{width}; int c=i%{width};
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}
    """,
        len(cases),
    )


def run_product(metadata, program, operands):
    """Execute uniform node products, falling back for shared/strided layouts."""
    width = parse_metadata(metadata)[0]
    nodes = operands[0].shape[0]
    if not width or any(x.shape[0] == 1 and nodes != 1 for x in operands):
        return reference_product(metadata, program, operands)
    outputs = channel_product_outputs(metadata, program, operands)
    if not nodes:
        for output in outputs:
            output.zero_()
        return outputs
    source, groups = product_source(metadata, program, operands[0].dtype, len(operands))
    kernel = kernels([source], operands[0].device)[source]
    arguments = [value.data_ptr() for value in operands]
    arguments += [stride for value in operands for stride in value.stride()]
    arguments += [value.data_ptr() for value in outputs] + [nodes]
    runtime().launch(
        [(kernel, arguments, (nodes * width + 127) // 128, groups, 128, 0)],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs


@torch.library.custom_op("eqx_oam_xl::product", mutates_args=(), device_types="cuda")
def product_contract(
    metadata: str, program: str, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate a complete channel product program with recursive adjoints."""
    return run_product(metadata, program, operands)


product_contract.register_fake(channel_product_outputs)


def setup_context(ctx, inputs, output):
    ctx.kernel_metadata, program, operands = inputs
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(*operands)


def backward(ctx, grad_outputs):
    operands = list(ctx.saved_tensors)
    values, cotangents = list(operands), {}
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
                terms.append(
                    (tuple(replacement), role, destination, required | (1 << output))
                )
    gradients = [None] * len(operands)
    if terms:
        results = product_contract(ctx.kernel_metadata, repr(tuple(terms)), values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, gradients


product_contract.register_autograd(backward, setup_context=setup_context)
