"""CUDA evaluation of symmetric polynomials with indexed coefficients."""

from collections import Counter, defaultdict
from functools import lru_cache

import torch

from ...ops.symmetric import symmetric_contract_fake
from ...utils.metadata import parse_metadata
from .runtime import kernels, runtime


@lru_cache(maxsize=128)
def polynomial_groups(metadata, program):
    """Combine permutation-equivalent products without combining learned paths."""
    _, _, _, degree, rows = parse_metadata(metadata)
    groups = defaultdict(lambda: defaultdict(float))
    for mapping, role, slot, required in parse_metadata(program):
        for indices, weight, output, coefficient in rows:
            indices = (*indices, *([-1] * (degree - len(indices))), weight, output)
            if indices[role] < 0 or any(
                index < 0 for k, index in enumerate(indices) if required & (1 << k)
            ):
                continue
            factors = tuple(
                sorted(
                    (mapping[k], index, k == degree)
                    for k, index in enumerate(indices)
                    if k != role and index >= 0
                )
            )
            groups[slot, role == degree, indices[role]][factors] += coefficient
    return tuple(
        (
            key,
            tuple(
                (factors, value) for factors, value in sorted(terms.items()) if value
            ),
        )
        for key, terms in sorted(groups.items())
    )


def polynomial_expression(terms, cache):
    """Factor sparse polynomials without materializing their monomials."""
    code = []

    def product(factors):
        if not factors:
            return "scalar(1)"
        if len(factors) > 1:
            return "*".join(product((factor,)) for factor in factors)
        if factors in cache:
            return cache[factors]
        operand, index, indexed = factors[0]
        value = f"p{operand}[{'z' if indexed else 'n'}*s{operand}+{index}*a{operand}+c*b{operand}]"
        name = f"v{len(cache)}"
        cache[factors] = name
        code.append(f"scalar {name}={value};")
        return name

    def expression(values):
        if not values:
            return "scalar(0)"
        counts = Counter(factor for factors, _ in values for factor in set(factors))
        if not counts or max(counts.values()) == 1:
            return "+".join(
                f"scalar({value:.17g})*{product(factors)}" for factors, value in values
            )
        factor = max(counts, key=lambda f: (counts[f], f[2], f))
        selected, remainder = [], []
        for factors, coefficient in values:
            if factor in factors:
                factors = list(factors)
                factors.remove(factor)
                selected.append((tuple(factors), coefficient))
            else:
                remainder.append((factors, coefficient))
        result = f"{product((factor,))}*({expression(selected)})"
        if remainder:
            result += f"+({expression(remainder)})"
        return result

    value = expression(terms)
    code.append(f"sum={value};")
    return "\n".join(code)


@lru_cache(maxsize=128)
def source(
    metadata, program, dtype, index_dtype, input_strides, channels, dimensions, reduce
):
    """Generate a single evaluation for each output or derivative entry."""
    cases = []
    groups = tuple(
        entry for entry in polynomial_groups(metadata, program) if entry[0][1] == reduce
    )
    batches, batch = [], []
    factors = set()
    for entry in groups:
        incoming = {factor for inputs, _ in entry[1] for factor in inputs}
        if batch and (
            entry[0][:2] != batch[0][0][:2]
            or len(batch) == (4 if entry[0][1] else 16)
            or len(factors | incoming) > 64
        ):
            batches.append(batch)
            batch = []
            factors = set()
        batch.append(entry)
        factors.update(incoming)
    if batch:
        batches.append(batch)
    for group, batch in enumerate(batches):
        cache, code = {}, []
        stores = []
        for entry, ((slot, indexed, offset), terms) in enumerate(batch):
            code.append("sum=0;")
            code.append(polynomial_expression(terms, cache))
            code.append(
                f"acc[{entry}]+=sum;"
                if indexed
                else f"o{slot}[(n*{dimensions[slot]}+{offset})*channels+c]=sum;"
            )
            stores.append(
                f"atomicAdd(o{slot}+(previous*{dimensions[slot]}+{offset})*channels+c,acc[{entry}]); acc[{entry}]=0;"
            )
        if reduce:
            code = [
                f"scalar acc[{len(batch)}]={{}}; long long previous=-1;",
                "for(int row=0;row<16 && start+row<nodes;++row) {",
                "long long n=order[start+row],z=types[n];",
                f"if(previous>=0 && z!=previous) {{ {''.join(stores)} }} previous=z;",
                *code,
                "}",
                f"if(previous>=0) {{ {''.join(stores)} }}",
            ]
        cases.append(f"case {group}: {{ scalar sum=0; {''.join(code)} break; }}")
    scalar = "double" if dtype == torch.float64 else "float"
    index = "long long" if index_dtype == torch.int64 else "int"
    pointers = ",".join(
        f"const scalar* __restrict__ p{k}" for k in range(len(input_strides))
    )
    strides = "\n".join(
        f"constexpr long long s{k}={s},a{k}={a},b{k}={b};"
        for k, (s, a, b) in enumerate(input_strides)
    )
    outputs = ",".join(f"scalar* o{k}" for k in range(len(dimensions)))
    position = (
        "long long start=(i/channels)*16; if(start>=nodes) return; int c=i%channels;"
        if reduce
        else "if(i>=nodes*channels) return; long long n=i/channels,z=types[n]; int c=i%channels;"
    )
    return (
        f"""
    using scalar={scalar};
    extern "C" __global__ void run({pointers},const {index}* types,const long long* order,
        {outputs},long long nodes) {{
        constexpr int channels={channels}; {strides}
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        {position}
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}
    """,
        len(batches),
    )


def contract(metadata, program, node_type, operands):
    """Evaluate the polynomial and arbitrary recursive derivatives."""
    if operands[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Symmetric CUDA contractions require float32 or float64.")
    outputs = symmetric_contract_fake(metadata, program, node_type, operands)
    groups = polynomial_groups(metadata, program)
    for slot, output in enumerate(outputs):
        assigned = {
            offset
            for (destination, indexed, offset), _ in groups
            if destination == slot and not indexed
        }
        if len(assigned) != output.shape[1]:
            output.zero_()
    nodes, channels = node_type.shape[0], operands[0].shape[-1]
    if not nodes:
        return outputs
    programs = [
        (
            *source(
                metadata,
                program,
                operands[0].dtype,
                node_type.dtype,
                tuple(tuple(x.stride()) for x in operands),
                channels,
                tuple(output.shape[1] for output in outputs),
                reduce,
            ),
            reduce,
        )
        for reduce in (False, True)
        if any(key[1] == reduce for key, _ in groups)
    ]
    order = (
        node_type.argsort() if any(reduce for _, _, reduce in programs) else node_type
    )
    args = [x.data_ptr() for x in operands]
    args += [
        node_type.data_ptr(),
        order.data_ptr(),
        *(x.data_ptr() for x in outputs),
        nodes,
    ]
    compiled = kernels([code for code, _, _ in programs], operands[0].device)
    launches = [
        (
            compiled[code],
            args,
            (((nodes + 15) // 16 if reduce else nodes) * channels + 127) // 128,
            count,
            128,
            0,
        )
        for code, count, reduce in programs
    ]
    if launches:
        runtime().launch(
            launches, torch.cuda.current_stream(operands[0].device).cuda_stream
        )
    return outputs
