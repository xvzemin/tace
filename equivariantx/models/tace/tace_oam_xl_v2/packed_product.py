"""CG, gate and irrep packing with recursive mixed derivatives.

The physical result stores each irrep's complete node matrix contiguously.
Its logical shape stays ``(nodes, features)``; :func:`packed_linear` consumes
this layout. Learned matrices retain their original identities.
"""

from .precision import require_unscaled, require_tensors

from collections import defaultdict
from functools import lru_cache
import os
import re

import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.ops.common.cuda.codegen import contraction_source
from equivariantx.utils.metadata import parse_metadata

POLYNOMIAL_PRODUCTS = os.environ.get("EQX_XL_POLYNOMIAL_PRODUCTS", "0")
POLYNOMIAL_PRODUCTS = (
    "auto" if POLYNOMIAL_PRODUCTS == "auto" else POLYNOMIAL_PRODUCTS == "1"
)


def packed_metadata(metadata, gate_index, gate_width, blocks):
    """Describe uniform-channel products and per-irrep physical blocks."""
    width, dims, rows = parse_metadata(metadata)
    if not width or len(dims) != 3 or gate_width % width:
        raise ValueError("Packed products require uniform channel multiplicities")
    gates = []
    for index in range(dims[2]):
        values = gate_index[index * width : (index + 1) * width]
        first = values[0]
        if first % width or values != list(range(first, first + width)):
            raise ValueError("Gate channels must be contiguous and aligned")
        gates.append(first // width)
    if sum(blocks) != dims[2]:
        raise ValueError("Irrep blocks do not cover the product result")
    return repr(
        (
            width,
            (*dims[:2], gate_width // width, dims[2]),
            tuple(blocks),
            tuple(((a, b, gates[z], z), c) for (a, b, z), c in rows),
        )
    )


def packed_outputs(metadata, program, operands):
    """Allocate logical result shapes; the contraction defines their layout."""
    width, dims, _, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (1 + max(slot for _, _, slot, _ in terms))
    for _, role, slot, _ in terms:
        if outputs[slot] is None:
            outputs[slot] = operands[0].new_empty(
                (operands[0].shape[0], dims[role] * width)
            )
    return outputs


@lru_cache(maxsize=128)
def packed_source(
    metadata, program, dtype, count, tile, polynomial=False, inline_factored=False
):
    """Generate bounded output tiles, sharing compact operand loads."""
    width, dims, blocks, rows = parse_metadata(metadata)
    terms = parse_metadata(program)
    block_map, begin = {}, 0
    for size in blocks:
        for index in range(begin, begin + size):
            block_map[index] = (begin, size, index - begin)
        begin += size

    def address(role, index, operand=None):
        if role == 3:
            start, size, offset = block_map[index]
            return f"nodes*{start * width}+n*{size * width}+{offset * width}+c"
        if operand is None:
            return f"n*{dims[role] * width}+{index * width}+c"
        return f"n*s{operand}+({index * width}+c)*t{operand}"

    groups, output_roles = defaultdict(list), {}
    for mapping, role, slot, required in terms:
        entries = defaultdict(list)
        for indices, coefficient in rows:
            if indices[role] < 0 or any(
                index < 0 and required & (1 << k) for k, index in enumerate(indices)
            ):
                continue
            entries[indices[role]].append((indices, coefficient))
        previous = output_roles.setdefault(slot, role)
        if dims[previous] != dims[role] or (previous == 3) != (role == 3):
            raise ValueError("A result slot must have one physical operand layout")
        for index in range(dims[role]):
            groups[slot, index].append((role, mapping, entries[index]))
    by_slot = defaultdict(list)
    for (slot, index), entries in sorted(groups.items()):
        by_slot[slot, output_roles[slot]].append((index, entries))
    cases = []
    for (slot, role), elements in by_slot.items():
        group_size = tile if role == 3 else min(tile, 4)
        for start in range(0, len(elements), group_size):
            chunk = elements[start : start + group_size]
            cached = {}
            for _, entries in chunk:
                for term_role, mapping, row in entries:
                    for indices, _ in row:
                        for k, index in enumerate(indices):
                            if k != term_role and k != 3 and index >= 0:
                                key = (mapping[k], index)
                                cached.setdefault(key, (k, f"v{len(cached)}"))
            declarations = [
                f"scalar {name}=p{operand}[{address(k, index, operand)}];"
                for (operand, index), (k, name) in cached.items()
            ]

            def value(k, index, mapping):
                if k == 3:
                    return f"p{mapping[k]}[{address(k, index)}]"
                return cached[mapping[k], index][1]

            bodies = []
            for index, entries in chunk:
                if inline_factored and role in (0, 1):
                    # Combine compatible gate/cotangent terms before the CG
                    # contraction without materializing a wide global array.
                    grouped = defaultdict(list)
                    for term_role, mapping, row in entries:
                        for indices, coefficient in row:
                            other = value(
                                1 - term_role, indices[1 - term_role], mapping
                            )
                            gate = value(2, indices[2], mapping)
                            cotangent = value(3, indices[3], mapping)
                            grouped[coefficient, other, indices[3]].append(
                                (1.0, (gate, cotangent))
                            )
                    monomials = []
                    for (coefficient, other, _), products in grouped.items():
                        weighted = contraction_source(products, cached, bodies)
                        monomials.append((coefficient, (other, weighted)))
                    result = contraction_source(monomials, cached, bodies)
                    bodies.append(f"o{slot}[{address(role, index)}]={result};")
                    continue
                if polynomial:
                    monomials = []
                    for term_role, mapping, row in entries:
                        for indices, coefficient in row:
                            factors = tuple(
                                value(k, feature, mapping)
                                for k, feature in enumerate(indices)
                                if k != term_role and feature >= 0
                            )
                            monomials.append((coefficient, factors))
                    result = contraction_source(monomials, cached, bodies)
                    bodies.append(f"o{slot}[{address(role, index)}]={result};")
                    continue
                body = ["{scalar total=0;"]
                for term_role, mapping, row in entries:
                    if term_role in (2, 3):
                        common_role = 2 if term_role == 3 else 3
                        partials = defaultdict(list)
                        for indices, coefficient in row:
                            partials[indices[common_role]].append(
                                (indices, coefficient)
                            )
                        body.append("{scalar part=0;")
                        for common, subset in partials.items():
                            body.append("{scalar cg=0;")
                            for indices, coefficient in subset:
                                a, b = (
                                    value(0, indices[0], mapping),
                                    value(1, indices[1], mapping),
                                )
                                body.append(
                                    f"cg+=scalar({coefficient:.17g})*({a}*{b});"
                                )
                            body.append(
                                f"part+=cg*{value(common_role, common, mapping)};}}"
                            )
                        body.append("total+=part;}")
                    else:
                        body.append("{scalar part=0;")
                        for indices, coefficient in row:
                            other = value(
                                1 - term_role, indices[1 - term_role], mapping
                            )
                            gate = value(2, indices[2], mapping)
                            cotangent = value(3, indices[3], mapping)
                            body.append(
                                f"part+=scalar({coefficient:.17g})*({other}*({gate}*{cotangent}));"
                            )
                        body.append("total+=part;}")
                body.append(f"o{slot}[{address(role, index)}]=total;}}")
                bodies.extend(body)
            cases.append(
                f"case {len(cases)}:{{{''.join(declarations + bodies)}break;}}"
            )
    pointers = ",".join(f"const scalar* p{i}" for i in range(count))
    strides = ",".join(f"long long s{i},long long t{i}" for i in range(count))
    outputs = ",".join(f"scalar* o{i}" for i in range(len(by_slot)))
    scalar = "double" if dtype == torch.float64 else "float"
    source = f"""
    using scalar={scalar};
    using T=scalar;
    extern "C" __global__ void run({pointers},{strides},{outputs},long long nodes) {{
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=nodes*{width}) return;
        long long n=i/{width}; int c=i%{width};
        switch(blockIdx.y) {{ {"".join(cases)} }}
    }}"""
    return source, len(cases)


def run_packed(metadata, program, operands, tile=16, fragments=(), output_scale=0):
    """Execute a complete value or adjoint program without unpacking."""
    require_unscaled(output_scale, "output_scale")
    require_tensors(operands)
    width, dimensions, _, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    sweep = os.environ.get("EQX_XL_NODE_SWEEP", "0") == "1"
    tangent = os.environ.get("EQX_XL_NODE_TANGENT", "0") == "1"
    if sweep or tangent:
        roles = {role for _, role, _, _ in terms}
        slot_roles = defaultdict(set)
        for _, role, slot, _ in terms:
            slot_roles[slot].add(role)
        if (
            ((sweep and roles == {0, 1, 2}) or (tangent and roles == {0, 3}))
            and dimensions[:2] == (25, 25)
            and all(len(value) == 1 for value in slot_roles.values())
        ):
            from .node_sweep import sweep_product

            # Full mixed terms need smaller live ranges; first reverse benefits
            # from grouping by the right input. Both retain the original paths.
            grouping = (
                "left"
                if roles == {0, 3} or (len(terms) > 3 and dimensions[3] > 79)
                else "right"
            )
            return sweep_product(
                metadata,
                program,
                operands,
                grouping,
                True,
                128,
                fragments,
                output_scale,
            )
    polynomial = POLYNOMIAL_PRODUCTS
    if polynomial == "auto":
        # Measured on every post-fusion production phase: factoring helps the
        # primal/tangent and the final l<=2 readout, but increases register
        # pressure in the full l<=4 reverse programs.
        output_roles = {role for _, role, _, _ in parse_metadata(program)}
        polynomial = 3 in output_roles or dimensions[3] <= 79
        if polynomial:
            tile = 32 if 3 in output_roles else 8
    roles = {mapping[3] for mapping, _, _, _ in parse_metadata(program)}
    operands = [
        value.contiguous() if i in roles else value for i, value in enumerate(operands)
    ]
    outputs = packed_outputs(metadata, program, operands)
    nodes = operands[0].shape[0]
    if not nodes:
        return outputs
    source, groups = packed_source(
        metadata,
        program,
        operands[0].dtype,
        len(operands),
        tile,
        polynomial,
        os.environ.get("EQX_XL_INLINE_PRODUCT_FACTORS", "0") == "1",
    )
    if fragments:
        from .packed_fragments import fragment_source

        source = fragment_source(source, metadata, fragments)
    args = [value.data_ptr() for value in operands]
    args += [stride for value in operands for stride in value.stride()]
    args += [value.data_ptr() for value in outputs] + [nodes]
    case_tile = int(os.environ.get("EQX_XL_PRODUCT_CASE_TILE", "0"))
    codes = (
        split_case_sources(source, groups, case_tile)
        if case_tile
        else [(source, groups)]
    )
    compiled = kernels([code for code, _ in codes], operands[0].device)
    runtime().launch(
        [
            (compiled[code], args, (nodes * width + 127) // 128, count, 128, 0)
            for code, count in codes
        ],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return outputs


@lru_cache(maxsize=128)
def split_case_sources(source, groups, tile):
    """Separate independent output cases so cheap cases use fewer registers."""
    if tile < 1:
        raise ValueError("Case tile must be positive")
    cases = list(re.finditer(r"case (\d+):\{", source))
    if len(cases) != groups:
        raise ValueError("Packed source does not match its declared case count")
    chunks = []
    for match in cases:
        depth, end = 1, match.end()
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        chunks.append(source[match.start() : end])
    prefix, suffix = source[: cases[0].start()], source[end:]
    result = []
    for begin in range(0, groups, tile):
        selected = chunks[begin : begin + tile]
        body = "".join(
            re.sub(r"^case \d+:", f"case {i}:", case) for i, case in enumerate(selected)
        )
        result.append((prefix + body + suffix, len(selected)))
    return result


@torch.library.custom_op(
    "eqx_oam_xl::packed_product", mutates_args=(), device_types="cuda"
)
def packed_contract(
    metadata: str, program: str, operands: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Evaluate a compact gated CG program and its recursive adjoints."""
    return run_packed(metadata, program, operands)


packed_contract.register_fake(packed_outputs)


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
        results = packed_contract(ctx.kernel_metadata, repr(tuple(terms)), values)
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, gradients


packed_contract.register_autograd(backward, setup_context=setup_context)


def packed_linear(linear, value):
    """Consume packed irrep matrices with the original optimized GEMMs."""
    flat, nodes = value.reshape(-1), value.shape[0]
    blocks = None
    if os.environ.get("EQX_XL_LAYOUT_ADJOINTS", "0") == "1":
        from .layout_adjoint import packed_slices

        blocks = packed_slices(value, linear.irreps_in)
    pieces = [[] for _ in linear.irreps_out]
    for matrix, instruction in zip(linear.weight, linear.linear.instructions):
        ci, ir = linear.irreps_in[instruction.i_in]
        co, _ = linear.irreps_out[instruction.i_out]
        offset = linear.irreps_in.slices()[instruction.i_in].start
        block = (
            blocks[instruction.i_in]
            if blocks is not None
            else flat.narrow(0, nodes * offset, nodes * ci * ir.dim).reshape(-1, ci)
        )
        result = (block @ matrix.reshape(ci, co)) * instruction.path_weight
        pieces[instruction.i_out].append(
            result.reshape(nodes, ir.dim, co).transpose(1, 2).reshape(nodes, -1)
        )
    outputs, bias_offset = [], 0
    for (co, ir), terms in zip(linear.irreps_out, pieces):
        if terms:
            result = terms[0]
            for term in terms[1:]:
                result = result + term
        else:
            result = value.new_zeros((nodes, co * ir.dim))
        if ir.is_scalar() and linear.bias is not None:
            result = result + linear.bias[bias_offset : bias_offset + co]
            bias_offset += co
        outputs.append(result)
    return torch.cat(outputs, dim=-1)


def composed_linear(first, last, value, *, packed=False, final_bias=False):
    """Associate consecutive equivariant matrices before the node contraction.

    ``(X A + a) B`` becomes ``X (A B) + a B``. Every original matrix and
    its normalization remains a live differentiable argument. No parameter
    is replaced or cached across optimizer steps.
    """
    nodes = value.shape[0]
    blocks = None
    if os.environ.get("EQX_XL_LAYOUT_ADJOINTS", "0") == "1":
        from .layout_adjoint import ColumnSlices, packed_slices

        blocks = (
            packed_slices(value, first.irreps_in)
            if packed
            else ColumnSlices.apply(
                value, tuple(c * ir.dim for c, ir in first.irreps_in)
            )
        )
    pieces = [[] for _ in last.irreps_out]
    first_bias = {}
    offset = 0
    for index, (channels, ir) in enumerate(first.irreps_out):
        if ir.is_scalar() and first.bias is not None:
            first_bias[index] = first.bias[offset : offset + channels]
            offset += channels
    for second_matrix, second in zip(last.weight, last.linear.instructions):
        mid, ir = last.irreps_in[second.i_in]
        co, _ = last.irreps_out[second.i_out]
        b = second_matrix.reshape(mid, co)
        for first_matrix, initial in zip(first.weight, first.linear.instructions):
            if initial.i_out != second.i_in:
                continue
            ci, input_ir = first.irreps_in[initial.i_in]
            if input_ir != ir:
                raise ValueError("Mismatched irreps in consecutive maps")
            start = first.irreps_in.slices()[initial.i_in].start
            if packed:
                x = (
                    blocks[initial.i_in]
                    if blocks is not None
                    else value.reshape(-1)
                    .narrow(0, nodes * start, nodes * ci * ir.dim)
                    .reshape(-1, ci)
                )
            else:
                x = (
                    blocks[initial.i_in]
                    if blocks is not None
                    else value[:, start : start + ci * ir.dim]
                )
                if getattr(first, "input_layout", "mul_ir") == "ir_mul":
                    x = x.reshape(nodes, ir.dim, ci)
                else:
                    x = x.reshape(nodes, ci, ir.dim).transpose(1, 2)
            combined = first_matrix.reshape(ci, mid) @ b
            term = (x @ combined) * (initial.path_weight * second.path_weight)
            pieces[second.i_out].append(
                term.reshape(nodes, ir.dim, co).transpose(1, 2).reshape(nodes, -1)
            )
        if second.i_in in first_bias:
            pieces[second.i_out].append(
                (first_bias[second.i_in] @ b) * second.path_weight
            )
    outputs = []
    offset = 0
    for (co, ir), terms in zip(last.irreps_out, pieces):
        result = value.new_zeros((nodes, co * ir.dim)) if not terms else terms[0]
        for term in terms[1:]:
            result = result + term
        if ir.is_scalar() and last.bias is not None:
            if final_bias:
                result = result + last.bias[offset : offset + co]
            offset += co
        outputs.append(result.expand(nodes, -1))
    return torch.cat(outputs, -1)
