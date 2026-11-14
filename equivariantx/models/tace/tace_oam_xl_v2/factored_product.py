"""Experimental exact gate factoring of packed node-product adjoints.

The original independent input factors and every CG row are retained. Shared
gate/cotangent products are materialized once and algebraically compatible
adjoints are summed before the bilinear CG contraction. No input value is
cached between invocations.
"""

from collections import defaultdict
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.pointwise import evaluate as pointwise
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata

from .packed_product import packed_outputs, packed_source, run_packed


@lru_cache(maxsize=128)
def factor_description(metadata, program):
    width, dims, blocks, rows = parse_metadata(metadata)
    terms = parse_metadata(program)
    gates = {}
    for indices, _ in rows:
        a, b, g, z = indices
        if a < 0 or b < 0 or g < 0 or z < 0:
            raise ValueError("Factoring expects a complete binary gated product")
        if z in gates and gates[z] != g:
            raise ValueError("Each CG output must have one gate")
        gates[z] = g
    if len(gates) != dims[3]:
        raise ValueError("Missing gate map")
    contractions = defaultdict(list)
    unchanged = []
    for mapping, role, slot, required in terms:
        if role in (0, 1):
            contractions[role, slot, mapping[1 - role]].append((mapping[2], mapping[3]))
        else:
            unchanged.append((mapping, role, slot, required))
    pair_sets = []
    new_terms = []
    # The retained operands have their original positions; new weighted
    # cotangents are appended. The absent gate has no derivative here.
    count = 1 + max(max(mapping) for mapping, _, _, _ in terms)
    for (role, slot, other), pairs in contractions.items():
        pairs = tuple(pairs)
        if len(pairs) > 2:
            raise ValueError("This experiment supports at most two fused cotangents")
        if pairs not in pair_sets:
            pair_sets.append(pairs)
        mapping = [other, other, 0, count + pair_sets.index(pairs)]
        new_terms.append((tuple(mapping), role, slot, 0))
    meta = repr(
        (width, dims, blocks, tuple(((a, b, -1, z), c) for (a, b, g, z), c in rows))
    )
    return (
        tuple(gates[z] for z in range(dims[3])),
        tuple(pair_sets),
        tuple(new_terms),
        tuple(unchanged),
        meta,
        count,
    )


@lru_cache(maxsize=64)
def gate_indices(metadata, program, device):
    return torch.tensor(
        factor_description(metadata, program)[0], device=device, dtype=torch.int32
    )


def dense_slots(terms):
    slots = tuple(dict.fromkeys(slot for _, _, slot, _ in terms))
    return slots, repr(
        tuple(
            (mapping, role, slots.index(slot), required)
            for mapping, role, slot, required in terms
        )
    )


def factored_product(metadata, program, operands, tile=4):
    """Evaluate a complete differentiated product after common gate factoring."""
    gates, pairs, contractions, unchanged, bilinear, count = factor_description(
        metadata, program
    )
    if not contractions:
        return run_packed(metadata, program, operands)
    width, dims, blocks, _ = parse_metadata(metadata)
    values = list(operands[:count])
    nodes = operands[0].shape[0]
    if not nodes:
        return packed_outputs(metadata, program, operands)
    index = gate_indices(metadata, program, values[0].device)
    for pair in pairs:
        g0, z0 = pair[0]
        g1, z1 = pair[1] if len(pair) == 2 else pair[0]
        first, second = values[z0].contiguous(), values[z1].contiguous()
        out = values[0].new_empty((nodes, dims[3] * width))
        code, first_column = [], 0
        for block in blocks:
            start, stop = (
                first_column * nodes * width,
                (first_column + block) * nodes * width,
            )
            code.append(
                f"""
            if(i>={start} && i<{stop}) {{
                I local=i-{start}, node=local/({block}*{width});
                I column=(local/{width})%{block}+{first_column};
                I gate=p4[column]*{width}+local%{width};
                T total=p0[node*{values[g0].stride(0)}+gate*{values[g0].stride(1)}]*p1[i];
                """
                + (
                    f"total += p2[node*{values[g1].stride(0)}+gate*{values[g1].stride(1)}]*p3[i];"
                    if len(pair) == 2
                    else ""
                )
                + "out[i]=total;}"
            )
            first_column += block
        pointwise(out, (values[g0], first, values[g1], second, index), "".join(code))
        values.append(out)
    destination, code = dense_slots(contractions)
    result = packed_outputs(bilinear, code, values)
    source, groups = packed_source(
        bilinear, code, values[0].dtype, len(values), tile, True
    )
    kernel = kernels([source], values[0].device)[source]
    arguments = [v.data_ptr() for v in values] + [s for v in values for s in v.stride()]
    arguments += [v.data_ptr() for v in result] + [nodes]
    runtime().launch(
        [(kernel, arguments, (nodes * width + 127) // 128, groups, 128, 0)],
        torch.cuda.current_stream(values[0].device).cuda_stream,
    )
    combined = [None] * (1 + max(slot for _, _, slot, _ in parse_metadata(program)))
    for slot, value in zip(destination, result):
        combined[slot] = value
    if unchanged:
        destination, code = dense_slots(unchanged)
        for slot, value in zip(destination, run_packed(metadata, code, operands)):
            if combined[slot] is not None:
                raise ValueError("Mixed destination roles are unsupported")
            combined[slot] = value
    return combined
