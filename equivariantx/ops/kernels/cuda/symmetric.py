"""CUDA evaluation of symmetric polynomials with indexed coefficients."""

from collections import Counter
from functools import lru_cache

import torch

from ....utils.metadata import parse_metadata
from ....utils.outputs import symmetric_outputs
from ...polynomial import polynomial_groups
from .runtime import kernels, runtime


def polynomial_expressions(expressions):
    """Share a bounded set of repeated subexpressions across sparse outputs."""
    nodes, lookup, costs = [], {}, []

    def node(kind, *args):
        key = kind, *args
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(key)
            costs.append(1 + sum(costs[i] for i in args) if kind in ("+", "*") else 0)
        return lookup[key]

    def combine(kind, values):
        values = iter(values)
        result = next(values, node("constant", 1.0 if kind == "*" else 0.0))
        for value in values:
            result = node(kind, result, value)
        return result

    def product(factors):
        return combine("*", (node("input", factor) for factor in factors))

    def expression(values):
        counts = Counter(factor for factors, _ in values for factor in set(factors))
        if not counts or max(counts.values()) == 1:
            return combine(
                "+",
                (
                    node("*", node("constant", value), product(factors))
                    for factors, value in values
                ),
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
        result = node("*", product((factor,)), expression(selected))
        return node("+", result, expression(remainder)) if remainder else result

    roots = [expression(terms) for terms in expressions]
    uses = Counter(roots)
    for kind, *args in nodes:
        if kind in ("+", "*"):
            uses.update(args)
    candidates = [i for i, cost in enumerate(costs) if cost >= 2 and uses[i] > 1]
    shared = set(
        sorted(candidates, key=lambda i: (uses[i] - 1) * costs[i], reverse=True)[:16]
    )
    cache, result = {}, []

    def emit(index, code):
        if index in cache:
            return cache[index]
        kind, *args = nodes[index]
        if kind == "constant":
            return f"scalar({args[0]:.17g})"
        if kind == "input":
            operand, component, indexed = args[0]
            value = f"p{operand}[{'z' if indexed else 'n'}*s{operand}+{component}*a{operand}+c*b{operand}]"
        else:
            value = f"({emit(args[0], code)}{kind}{emit(args[1], code)})"
        if kind == "input" or index in shared:
            name = f"v{len(cache)}"
            cache[index] = name
            code.append(f"scalar {name}={value};")
            return name
        return value

    for root in roots:
        code = []
        value = emit(root, code)
        code.append(f"sum={value};")
        result.append("\n".join(code))
    return result


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
def polynomial_batches(metadata, program, reduce):
    """Bound the input and output working set of each contraction packet."""
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
    return batches


@lru_cache(maxsize=128)
def reduction_rows(nodes, channels, batches, device):
    """Keep small parameter-gradient grids parallel without extra large-grid atomics."""
    rows = 16
    blocks = 2 * torch.cuda.get_device_properties(device).multi_processor_count
    while (
        rows > 4
        and ((nodes + rows - 1) // rows * channels + 127) // 128 * batches < blocks
    ):
        rows //= 2
    return rows


@lru_cache(maxsize=128)
def source(
    metadata,
    program,
    dtype,
    index_dtype,
    input_strides,
    channels,
    dimensions,
    reduce,
    rows=16,
):
    """Generate a single evaluation for each output or derivative entry."""
    cases = []
    batches = polynomial_batches(metadata, program, reduce)
    degree = parse_metadata(metadata)[3]
    for group, batch in enumerate(batches):
        code = []
        stores = []
        if dtype == torch.float64 and degree >= 3:
            expressions = polynomial_expressions([terms for _, terms in batch])
        else:
            cache = {}
            expressions = [polynomial_expression(terms, cache) for _, terms in batch]
        for entry, ((slot, indexed, offset), terms) in enumerate(batch):
            code.append("sum=0;")
            code.append(expressions[entry])
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
                f"for(int row=0;row<{rows} && start+row<nodes;++row) {{",
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
        f"long long start=(i/channels)*{rows}; if(start>=nodes) return; int c=i%channels;"
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
    outputs = symmetric_outputs(metadata, program, node_type, operands)
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
    programs = []
    for reduce in (False, True):
        batches = polynomial_batches(metadata, program, reduce)
        if not batches:
            continue
        rows = (
            reduction_rows(nodes, channels, len(batches), operands[0].device)
            if reduce
            else 16
        )
        code, count = source(
            metadata,
            program,
            operands[0].dtype,
            node_type.dtype,
            tuple(tuple(x.stride()) for x in operands),
            channels,
            tuple(output.shape[1] for output in outputs),
            reduce,
            rows,
        )
        programs.append((code, count, reduce, rows))
    order = (
        node_type.argsort()
        if any(reduce for _, _, reduce, _ in programs)
        else node_type
    )
    args = [x.data_ptr() for x in operands]
    args += [
        node_type.data_ptr(),
        order.data_ptr(),
        *(x.data_ptr() for x in outputs),
        nodes,
    ]
    compiled = kernels([code for code, _, _, _ in programs], operands[0].device)
    launches = [
        (
            compiled[code],
            args,
            (((nodes + rows - 1) // rows if reduce else nodes) * channels + 127) // 128,
            count,
            128,
            0,
        )
        for code, count, reduce, rows in programs
    ]
    if launches:
        runtime().launch(
            launches, torch.cuda.current_stream(operands[0].device).cuda_stream
        )
    return outputs
