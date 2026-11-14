"""Share cotangent loads across all node-product adjoint outputs.

A thread owns one node/channel and streams complete CG paths. It accumulates
input adjoints in registers and writes each gate adjoint at the end of its
path. Unlike separate output tiles, input and gate gradients reuse the same
loaded cotangents. This is an opt-in component experiment after autodiff.
"""

from .precision import require_unscaled, require_tensors
from collections import defaultdict
from functools import lru_cache
import math

import torch

from equivariantx.ops.common.cuda.codegen import contraction_source
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from .packed_product import packed_outputs


@lru_cache(maxsize=128)
def sweep_sources(metadata, program, dtype, count, grouping, polynomial):
    width, dims, blocks, rows = parse_metadata(metadata)
    terms = parse_metadata(program)
    if any(role not in (0, 1, 2, 3) for _, role, _, _ in terms):
        raise ValueError("Node sweeps require explicit binary gated products")
    if any(min(indices) < 0 for indices, _ in rows):
        raise ValueError("Node sweeps require complete binary products")
    by_gate = defaultdict(list)
    for row in rows:
        by_gate[row[0][2]].append(row)
    if any(role == 3 for _, role, _, _ in terms):
        output_gates = defaultdict(set)
        for indices, _ in rows:
            output_gates[indices[3]].add(indices[2])
        if any(len(gates) > 1 for gates in output_gates.values()):
            raise ValueError("Forward sweep requires disjoint output paths")
    groups = defaultdict(list)
    for gate, entries in sorted(by_gate.items()):
        if grouping == "all":
            key = 0
        elif grouping in ("left", "right"):
            axis = 0 if grouping == "left" else 1
            degrees = {math.isqrt(indices[axis]) for indices, _ in entries}
            if len(degrees) != 1:
                raise ValueError("Expected one input degree per gate path")
            key = next(iter(degrees))
        else:
            key = gate // int(grouping)
        groups[key].append(gate)
    destinations = {slot: role for _, role, slot, _ in terms}
    if any(destinations[slot] != role for _, role, slot, _ in terms):
        raise ValueError("One output slot must have one layout")
    owners = defaultdict(set)
    for group, gates in groups.items():
        for gate in gates:
            for _, role, slot, _ in terms:
                for indices, _ in by_gate[gate]:
                    owners[slot, indices[role]].add(group)
    packed, start = {}, 0
    for block in blocks:
        for index in range(start, start + block):
            packed[index] = start, block, index - start
        start += block
    scalar = "double" if dtype == torch.float64 else "float"
    pointers = ",".join(f"const T* p{i}" for i in range(count))
    strides = ",".join(f"long long s{i},long long t{i}" for i in range(count))
    outputs = ",".join(f"T* o{i}" for i in range(1 + max(destinations)))

    def input_address(role, index, operand):
        if role == 3:
            begin, length, local = packed[index]
            return f"nodes*{begin * width}+n*{length * width}+{local * width}+c"
        return f"n*s{operand}+({index * width}+c)*t{operand}"

    def output_address(role, index):
        if role == 3:
            begin, length, local = packed[index]
            return f"nodes*{begin * width}+n*{length * width}+{local * width}+c"
        return f"n*{dims[role] * width}+{index * width}+c"

    sources = []
    atomic_slots = {slot for (slot, _), groups_ in owners.items() if len(groups_) > 1}
    for group, gates in groups.items():
        active = sorted(
            (slot, index)
            for (slot, index), members in owners.items()
            if group in members and destinations[slot] in (0, 1)
        )
        lines = [
            f"using T={scalar};",
            f'extern "C" __global__ void run({pointers},{strides},{outputs},long long nodes) {{',
            "long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;",
            f"if(i>=nodes*{width}) return;",
            f"long long n=i/{width}; int c=i%{width};",
        ]
        lines.extend(f"T acc_{slot}_{index}=0;" for slot, index in active)
        for gate in gates:
            lines.append("{")
            cache, expressions = {}, []
            values = {}
            for mapping, role, slot, required in terms:
                for indices, _ in by_gate[gate]:
                    for other in range(4):
                        if other == role:
                            continue
                        key = mapping[other], indices[other]
                        if key not in values:
                            name = f"input_{len(values)}"
                            values[key] = name
                            lines.append(
                                f"const T {name}=p{key[0]}[{input_address(other, key[1], key[0])}];"
                            )
            sums = defaultdict(list)
            for mapping, role, slot, required in terms:
                for indices, coefficient in by_gate[gate]:
                    factors = tuple(
                        values[mapping[other], indices[other]]
                        for other in range(4)
                        if other != role
                    )
                    sums[slot, indices[role]].append((coefficient, factors))
            for (slot, index), monomials in sorted(sums.items()):
                if polynomial:
                    expression = contraction_source(monomials, cache, expressions)
                    lines.extend(expressions)
                    expressions.clear()
                else:
                    expression = f"part_{slot}_{index}"
                    lines.append(f"T {expression}=0;")
                    lines.extend(
                        f"{expression} += T({coef:.17g})*({'*'.join(factors)});"
                        for coef, factors in monomials
                    )
                role = destinations[slot]
                if role in (0, 1):
                    lines.append(f"acc_{slot}_{index} += {expression};")
                else:
                    lines.append(
                        f"o{slot}[{output_address(role, index)}]={expression};"
                    )
            lines.append("}")
        for slot, index in active:
            address = f"o{slot}+({output_address(destinations[slot], index)})"
            value = f"acc_{slot}_{index}"
            lines.append(
                f"atomicAdd({address},{value});"
                if len(owners[slot, index]) > 1
                else f"*({address})={value};"
            )
        lines.append("}")
        sources.append("\n".join(lines))
    return tuple(sources), tuple(sorted(atomic_slots))


def sweep_product(
    metadata,
    program,
    operands,
    grouping="all",
    polynomial=True,
    threads=128,
    fragments=(),
    output_scale=0,
):
    """Evaluate a complete explicit node-adjoint program with shared loads."""
    require_unscaled(output_scale, "output_scale")
    require_tensors(operands)
    width, dims, _, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    packed_operands = {mapping[3] for mapping, _, _, _ in terms}
    operands = [
        value.contiguous() if index in packed_operands else value
        for index, value in enumerate(operands)
    ]
    outputs = packed_outputs(metadata, program, operands)
    nodes = operands[0].shape[0]
    if not nodes:
        return outputs
    sources, atomic_slots = sweep_sources(
        metadata, program, operands[0].dtype, len(operands), grouping, polynomial
    )
    if fragments:
        from .packed_fragments import fragment_source

        sources = tuple(
            fragment_source(source, metadata, fragments) for source in sources
        )
    for slot in atomic_slots:
        outputs[slot].zero_()
    compiled = kernels(list(sources), operands[0].device)
    arguments = [value.data_ptr() for value in operands]
    arguments += [stride for value in operands for stride in value.stride()]
    arguments += [value.data_ptr() for value in outputs] + [nodes]
    runtime().launch(
        [
            (
                compiled[source],
                arguments,
                (nodes * width + threads - 1) // threads,
                1,
                threads,
                0,
            )
            for source in sources
        ],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs
