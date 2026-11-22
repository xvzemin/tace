"""CG, gate and irrep packing with recursive mixed derivatives.

The physical result stores each irrep's complete node matrix contiguously.
Its logical shape stays ``(nodes, features)``; :func:`packed_linear` consumes
this layout. Learned matrices retain their original identities.
"""

from .precision import require_unscaled, require_tensors

from collections import defaultdict
from functools import lru_cache
from itertools import accumulate, groupby
import os
import re

import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.ops.common.cuda.codegen import contraction_source
from equivariantx.utils.metadata import compact_metadata, numeric_rows, parse_metadata

POLYNOMIAL_PRODUCTS = os.environ.get("EQX_XL_POLYNOMIAL_PRODUCTS", "0")
POLYNOMIAL_PRODUCTS = (
    "auto" if POLYNOMIAL_PRODUCTS == "auto" else POLYNOMIAL_PRODUCTS == "1"
)


SPARSE_ROWS = 12000


def angular_paths(product, gates, channels):
    """Describe each independently weighted path in the simplified ir_mul layout."""
    layouts = []
    for irreps in (product.irreps_in1, product.irreps_in2, product.irreps_out):
        layout, offset = [], 0
        for ir, group in groupby(irreps, key=lambda block: block.ir):
            copies = tuple(group)
            if any(mul != channels for mul, _ in copies):
                raise ValueError(
                    "Expected equal multiplicities for a packed uuu product"
                )
            stride = len(copies)
            layout.extend((ir.l, offset + i, stride) for i in range(stride))
            offset += stride * ir.dim
        layouts.append(layout)
    result = []
    for path in product.instructions:
        if not path.path_weight:
            continue
        a = layouts[0][path.i_in1]
        b = layouts[1][path.i_in2]
        l, base, stride = layouts[2][path.i_out]
        gate = gates[base]
        if any(gates[base + m * stride] != gate for m in range(1, 2 * l + 1)):
            return None
        result.append((*a, *b, l, base, stride, gate, path.path_weight))
    return tuple(result)


def gate_components(gate_index, width, size):
    """Map contiguous per-channel coefficients to their angular components."""
    index = torch.as_tensor(gate_index, dtype=torch.long, device="cpu").reshape(
        size, width
    )
    first = index[:, 0]
    if (first % width).any() or not torch.equal(
        index, first[:, None] + torch.arange(width, device="cpu")
    ):
        raise ValueError("Gate channels must be contiguous and aligned")
    return (first // width).tolist()


def angular_metadata(product, gate_index, gate_width, blocks, width):
    """Build a large packed contraction directly from its angular instructions."""
    from ..nonlinear_ace import spherical_rows

    irreps = (product.irreps_in1, product.irreps_in2, product.irreps_out)
    count = 0
    for path in product.instructions:
        if path.path_weight:
            count += len(
                spherical_rows(
                    irreps[0][path.i_in1].ir.l,
                    irreps[1][path.i_in2].ir.l,
                    irreps[2][path.i_out].ir.l,
                )
            )
        if count >= SPARSE_ROWS:
            break
    else:
        return None
    dims = tuple(sum(ir.dim for _, ir in group) for group in irreps)
    if sum(blocks) != dims[2] or gate_width % width:
        raise ValueError("Packed angular blocks do not match the gate layout")
    gates = gate_components(gate_index, width, dims[2])
    paths = angular_paths(product, gates, width)
    if paths is None:
        return None
    dimensions = (*dims[:2], gate_width // width, dims[2])
    return compact_metadata(repr((width, dimensions, tuple(blocks), ("wigner", paths))))


def packed_metadata(metadata, gate_index, gate_width, blocks):
    """Describe uniform-channel products and per-irrep physical blocks."""
    width, dims, rows = parse_metadata(metadata)
    if not width or len(dims) != 3 or gate_width % width:
        raise ValueError("Packed products require uniform channel multiplicities")
    if sum(blocks) != dims[2]:
        raise ValueError("Irrep blocks do not cover the product result")
    gates = gate_components(gate_index, width, dims[2])
    dimensions = (*dims[:2], gate_width // width, dims[2])
    return numeric_rows(
        width,
        dimensions,
        tuple(blocks),
        tuple(((a, b, gates[z], z), c) for (a, b, z), c in rows),
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


@lru_cache(maxsize=8)
def sparse_rows(metadata):
    """Expand tensor-free angular instructions into an immutable CPU coefficient table."""
    import numpy as np

    from equivariantx.models.tace.nonlinear_ace import spherical_rows

    _, _, _, rows = parse_metadata(metadata)
    if rows[0] != "wigner":
        indices, coefficients = zip(*rows)
        return np.asarray(indices, dtype=np.int32), np.asarray(
            coefficients, dtype=np.float64
        )
    paths = rows[1]
    count = sum(len(spherical_rows(a, b, c)) for a, _, _, b, _, _, c, *_ in paths)
    indices = np.empty((count, 4), dtype=np.int32)
    weights = np.empty(count, dtype=np.float64)
    start = 0
    for l1, a, sa, l2, b, sb, l3, z, sz, gate, factor in paths:
        local = np.asarray(spherical_rows(l1, l2, l3), dtype=np.float64)
        stop = start + len(local)
        for col, offset, stride, values in (
            (0, a, sa, 0),
            (1, b, sb, 1),
            (3, z, sz, 2),
        ):
            indices[start:stop, col] = offset + stride * local[:, values]
        indices[start:stop, 2] = gate
        weights[start:stop] = factor * local[:, 3]
        start = stop
    return indices, weights


@lru_cache(maxsize=128)
def sparse_product_table(metadata, role, dtype, device):
    """Index the CG rows by the output component, preserving each path's sum order."""
    import numpy as np

    _, dims, blocks, _ = parse_metadata(metadata)
    entries, weights = sparse_rows(metadata)
    columns = entries[:, role]
    order = np.argsort(columns, kind="stable")
    offsets = np.empty(dims[role] + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(np.bincount(columns, minlength=dims[role]), out=offsets[1:])
    sizes = np.asarray(blocks, dtype=np.int32)
    starts = np.cumsum(sizes, dtype=np.int32) - sizes
    block_info = np.repeat(np.stack((starts, sizes), axis=-1), sizes, axis=0)
    return (
        torch.from_numpy(offsets).to(device=device),
        torch.from_numpy(entries[order]).to(device=device),
        torch.from_numpy(weights[order]).to(device=device, dtype=dtype),
        torch.from_numpy(block_info).to(device=device),
    )


@lru_cache(maxsize=96)
def sparse_product_source(dtype, role, starts=()):
    """Generate a channel-coalesced sparse value or adjoint kernel."""
    scalar = "double" if dtype == torch.float64 else "float"
    pointers = "".join(f", const T* piece{i}" for i in range(len(starts)))
    packed_read = "pv=p[((I)z.x*nodes+n*z.y+e.w-z.x)*width+c];"
    if starts:
        packed_read = (
            "I local=(n*z.y+e.w-z.x)*width+c; switch(z.x){"
            + "".join(
                f"case {start}: pv=piece{i}[local]; break;"
                for i, start in enumerate(starts)
            )
            + "}"
        )
    return (
        """
    struct alignas(16) Row {int x,y,z,w;};
    struct alignas(8) Block {int x,y;};
    using I=long long;
    using T="""
        + scalar
        + f""";
    constexpr int role={role};
    extern "C" __global__ void run(
        const T* a,const T* b,const T* g,const T* p,
        I sa,I ta,I sb,I tb,I sg,I tg,const int* ptr,const Row* rows,
        const T* weights,const Block* blocks,T* out,I nodes,I width,I dim,I add{pointers}) {{
        I lane=(I)blockIdx.y*blockDim.x+threadIdx.x;
        I idx=blockIdx.x;
        if(lane>=nodes*width) return;
        I n=lane/width,c=lane%width;
        T v=0;
        for(int j=ptr[idx],end=ptr[idx+1];j<end;++j) {{
            Row e=rows[j];
            T av=1,bv=1,gv=1,pv=1;
            if constexpr(role!=0) av=a[n*sa+(e.x*width+c)*ta];
            if constexpr(role!=1) bv=b[n*sb+(e.y*width+c)*tb];
            if constexpr(role!=2) gv=g[n*sg+(e.z*width+c)*tg];
            if constexpr(role!=3) {{
                Block z=blocks[e.w];
                {packed_read}
            }}
            v+=weights[j]*((av*bv)*(gv*pv));
        }}
        I address=(n*dim+idx)*width+c;
        if constexpr(role==3) {{
            Block z=blocks[idx];
            address=((I)z.x*nodes+n*z.y+idx-z.x)*width+c;
        }}
        if(add) out[address]+=v; else out[address]=v;
    }}"""
    )


def sparse_packed_product(metadata, terms, operands, outputs, fragments):
    """Execute all terms with shared sparse tables and no atomic reductions."""
    from equivariantx.ops.common.cuda.graph import capture_constant

    width, dims, blocks, _ = parse_metadata(metadata)
    sample = operands[0]
    nodes = sample.shape[0]
    parts = dict(fragments)
    starts = tuple(accumulate((0, *blocks[:-1]))) if parts else ()
    sources = {
        (role, mapping[3]): sparse_product_source(
            sample.dtype, role, starts if mapping[3] in parts and role != 3 else ()
        )
        for mapping, role, _, _ in terms
    }
    compiled = kernels(tuple(sources.values()), sample.device)
    launches, written = [], set()
    for mapping, role, slot, _ in terms:
        table = capture_constant(
            sparse_product_table, metadata, role, sample.dtype, sample.device
        )
        values = [operands[i] for i in mapping]
        args = [v.data_ptr() for v in values]
        args.extend(stride for value in values[:3] for stride in value.stride())
        args.extend(x.data_ptr() for x in table)
        args.extend(
            (outputs[slot].data_ptr(), nodes, width, dims[role], int(slot in written))
        )
        if role != 3:
            args.extend(operands[i].data_ptr() for i in parts.get(mapping[3], ()))
        launches.append(
            (
                compiled[sources[role, mapping[3]]],
                args,
                dims[role],
                (nodes * width + 127) // 128,
                128,
                0,
            )
        )
        written.add(slot)
    runtime().launch(launches, torch.cuda.current_stream(sample.device).cuda_stream)
    return outputs


@lru_cache(maxsize=128)
def use_sparse_product(metadata):
    rows = parse_metadata(metadata)[3]
    return bool(rows) and (
        rows[0] == "wigner"
        or len(rows) >= SPARSE_ROWS
        and all(min(indices) >= 0 for indices, _ in rows)
    )


def run_packed(metadata, program, operands, tile=16, fragments=(), output_scale=0):
    """Execute a complete value or adjoint program without unpacking."""
    require_unscaled(output_scale, "output_scale")
    require_tensors(operands)
    width, dimensions, _, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    sweep = os.environ.get("EQX_XL_NODE_SWEEP", "0") == "1"
    tangent = os.environ.get("EQX_XL_NODE_TANGENT", "0") == "1"
    if (sweep or tangent) and not use_sparse_product(metadata):
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
    if use_sparse_product(metadata):
        return sparse_packed_product(metadata, terms, operands, outputs, fragments)
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
