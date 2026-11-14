"""CUDA evaluation of symmetric polynomials with indexed coefficients."""

import hashlib
import linecache
from collections import Counter
from functools import lru_cache

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ....utils.metadata import parse_metadata
from ....utils.outputs import symmetric_outputs
from ... import config
from ..polynomial import polynomial_groups, polynomial_prefix
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
    node_type = node_type.contiguous()
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


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def compiled_packets(metadata, program, budget):
    """Split complete term lists until every generated kernel fits the budget."""

    def split(packet):
        if len(polynomial_prefix(packet)[0]) <= budget:
            return (packet,)
        if len(packet) > 1:
            middle = len(packet) // 2
            parts = (packet[:middle], packet[middle:])
        else:
            key, terms = packet[0]
            if len(terms) <= 1:
                return None
            middle = len(terms) // 2
            parts = (((key, terms[:middle]),), ((key, terms[middle:]),))
        result = []
        for part in parts:
            children = split(part)
            if children is None:
                return None
            result.extend(children)
        return tuple(result)

    result = []
    for reduce in (False, True):
        packets = []
        for packet in polynomial_batches(metadata, program, reduce):
            pieces = split(tuple(packet))
            if pieces is None:
                return None
            packets.extend(pieces)
        chunks, current, size = [], [], 0
        for packet in packets:
            cost = len(polynomial_prefix(packet)[0])
            if current and size + cost > budget:
                chunks.append(tuple(current))
                current, size = [], 0
            current.append(packet)
            size += cost
        if current:
            chunks.append(tuple(current))
        result.append(tuple(chunks))
    return tuple(result)


@triton.jit
def segment_sum(left, left_type, right, right_type):
    return tl.where(left_type == right_type, left + right, right), right_type


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def compiled_kernel(packets, num_operands, dimensions, channels, reduce, accumulate=()):
    """Generate bounded Horner expressions with indexed segmented reductions."""
    arguments = [f"p{i}" for i in range(num_operands)]
    strides = [f"p{i}s{axis}" for i in range(num_operands) for axis in range(3)]
    outputs = [f"o{i}" for i in range(len(dimensions))]
    signature = [
        *arguments,
        *strides,
        "types",
        "order",
        *outputs,
        "count",
        "BR: tl.constexpr",
        "BC: tl.constexpr",
    ]
    lines = ["def polynomial(" + ", ".join(signature) + "):"]
    if reduce:
        lines += [
            "    row = tl.program_id(0) * BR + tl.arange(0, BR)",
            "    c = tl.program_id(1) * BC + tl.arange(0, BC)",
            "    n = tl.load(order + row, row < count, 0)",
            "    z = tl.load(types + n, row < count, -1)",
            "    next_n = tl.load(order + row + 1, row + 1 < count, 0)",
            "    next_z = tl.load(types + next_n, row + 1 < count, -1)",
            "    last = (next_z != z) | (row % BR == BR - 1) | (row == count - 1)",
            f"    mask = (row[:, None] < count) & (c[None, :] < {channels})",
            "    n = n[:, None]",
            "    z = z[:, None]",
            "    c = c[None, :]",
            "    z_grid = tl.broadcast_to(z, (BR, BC))",
        ]
    else:
        lines += [
            "    index = tl.program_id(0) * BR + tl.arange(0, BR)",
            f"    n = index // {channels}",
            f"    c = index % {channels}",
            "    mask = n < count",
            "    z = tl.load(types + n, mask, 0)",
        ]
    for packet, groups in enumerate(packets):
        indentation = "    "
        if len(packets) > 1:
            lines.append(f"    if tl.program_id(2) == {packet}:")
            indentation += "    "
        nodes, roots = polynomial_prefix(groups)
        for index, (operation, *args) in enumerate(nodes):
            if operation == "constant":
                value = f"tl.full((), {args[0]!r}, p0.dtype.element_ty)"
            elif operation == "input":
                operand, component, indexed = args
                value = (
                    f"tl.load(p{operand} + {'z' if indexed else 'n'} * p{operand}s0"
                    f" + {component} * p{operand}s1 + c * p{operand}s2, mask, 0)"
                )
            else:
                value = f"v{args[0]} {'+' if operation == 'add' else '*'} v{args[1]}"
            lines.append(f"{indentation}v{index} = {value}")
        for entry, ((slot, _, component), root) in enumerate(roots):
            destination = (
                f"o{slot} + ({'z' if reduce else 'n'} * {dimensions[slot]}"
                f" + {component}) * {channels} + c"
            )
            if reduce:
                lines += [
                    f"{indentation}v_reduce{entry}, _ = tl.associative_scan("
                    f"(tl.broadcast_to(v{root}, (BR, BC)), z_grid), 0, segment_sum)",
                    f"{indentation}tl.atomic_add({destination}, v_reduce{entry}, "
                    'mask & last[:, None], sem="relaxed")',
                ]
            elif (slot, component) in accumulate:
                lines.append(
                    f'{indentation}tl.atomic_add({destination}, v{root}, mask, sem="relaxed")'
                )
            else:
                lines.append(f"{indentation}tl.store({destination}, v{root}, mask)")
    source = "\n".join(lines) + "\n"
    name = "<eqx_symmetric_" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace = {"tl": tl, "segment_sum": segment_sum, "__name__": __name__}
    exec(compile(source, name, "exec"), namespace)
    return triton.jit(namespace["polynomial"])


def stride_primitive():
    """Return the optional installed primitive without changing global policy."""
    try:
        from torch._inductor.inductor_prims import force_stride_order
    except ImportError:
        return None
    return force_stride_order


def materialize(value):
    """Share captured contiguous inputs; otherwise retain the original tensor."""
    if not torch.compiler.is_compiling() and get_proxy_mode() is None:
        return value
    if not value.is_contiguous():
        return value
    operation = stride_primitive()
    return value if operation is None else operation(value, value.stride())


@triton_op("eqx::symmetric_compiled", mutates_args=())
def compiled_contract(
    metadata: str,
    program: str,
    node_type: torch.Tensor,
    operands: list[torch.Tensor],
    budget: int,
) -> list[torch.Tensor]:
    """Retain native polynomial kernels in the surrounding Inductor graph."""
    node_type = node_type.contiguous()
    groups = compiled_packets(metadata, program, budget)
    if groups is None:
        raise ValueError("The polynomial exceeds the native kernel budget.")
    operands = [materialize(value) for value in operands]
    node_type = materialize(node_type)
    outputs = symmetric_outputs(metadata, program, node_type, operands)
    assigned = {
        (slot, component)
        for packets in groups[0]
        for packet in packets
        for (slot, _, component), _ in packet
    }
    writers = Counter(
        (slot, component)
        for chunks in groups
        for packets in chunks
        for packet in packets
        for (slot, _, component), _ in packet
    )
    accumulate = tuple(key for key, count in writers.items() if count > 1)
    for slot, output in enumerate(outputs):
        if sum(key[0] == slot for key in assigned) != output.shape[1] or any(
            key[0] == slot for key in accumulate
        ):
            output.zero_()
    rows, channels = node_type.shape[0], operands[0].shape[-1]
    if rows and channels:
        order = node_type.argsort() if groups[1] else node_type
        strides = tuple(stride for value in operands for stride in value.stride())
        dimensions = tuple(value.shape[1] for value in outputs)
        for reduce, chunks in enumerate(groups):
            for packets in chunks:
                kernel = compiled_kernel(
                    packets,
                    len(operands),
                    dimensions,
                    channels,
                    bool(reduce),
                    accumulate,
                )
                if reduce:
                    grid = (triton.cdiv(rows, 16), triton.cdiv(channels, 32))
                    br, bc = 16, 32
                else:
                    grid = (triton.cdiv(rows * channels, 128), 1)
                    br, bc = 128, 1
                if len(packets) > 1:
                    grid = (*grid, len(packets))
                wrap_triton(kernel)[grid](
                    *operands,
                    *strides,
                    node_type,
                    order,
                    *outputs,
                    rows,
                    br,
                    bc,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
    return outputs


def compiled_evaluate(metadata, program, node_type, operands):
    """Use bounded native kernels, with ordinary Torch for larger derivatives."""
    if not all(value.is_cuda for value in operands):
        raise ValueError("Native symmetric polynomials require CUDA tensors.")
    budget = config.SYMMETRIC_NATIVE_MAX_NODES
    if compiled_packets(metadata, program, budget) is None:
        from ..torch.symmetric import evaluate_torch

        return evaluate_torch(metadata, program, node_type, operands)
    return compiled_contract(metadata, program, node_type, list(operands), budget)
