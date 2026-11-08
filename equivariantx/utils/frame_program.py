"""Local-frame rotation expressions."""

import math

from ..o2.layout import wigner_indices, wigner_orders


def frame_description(frame):
    return (
        frame.mmax,
        frame.basis_change,
        tuple((ir.dim, mul) for ir, mul in frame.irreps_out),
        tuple(
            (
                entry.global_slice.start,
                entry.mul,
                entry.degree,
                entry.odd,
                entry.local_indices,
                tuple(s.start for s in entry.local_slices),
            )
            for entry in frame._entries
        ),
    )


def rotate(
    program,
    description,
    features,
    wigner,
    local_dim,
    global_dim,
    inverse=False,
    packed=False,
    native=False,
):
    """Encode degree-wise rotation, regrouping and signed basis permutations."""
    mmax, basis_change, irreps, entries = description
    if packed:
        lmax = wigner_mmax = max(x[2] for x in entries) if entries else 0
    else:
        lmax, wigner_mmax = wigner_orders(
            global_dim,
            local_dim,
            lmax=max(x[2] for x in entries) if entries else 0,
            mmax=mmax,
        )
    if native:
        return indexed_rotation(
            program, description, features, wigner, local_dim, global_dim,
            lmax, wigner_mmax, inverse, packed,
        )
    slices, offset = [], 0
    for dim, mul in irreps:
        slices.append(offset)
        offset += dim * mul
    parts = [[] for _ in irreps]
    outputs = []
    for start, mul, ell, odd, indices, starts in entries:
        retained = min(ell, mmax)
        dim = 2 * ell + 1
        rows = (
            (ell, *(ell + s * m for m in range(1, retained + 1) for s in (1, -1)))
            if packed
            else wigner_indices(ell, retained, lmax)
        )
        offset = ell * (4 * ell**2 - 1) // 3
        if inverse:
            values = []
            for m, (index, begin) in enumerate(zip(indices, starts)):
                width = irreps[index][1]
                for real in range(1 if m == 0 else 2):
                    swapped = basis_change and odd and m > 0
                    position = (1 - real) if swapped else real
                    value = program.slice(
                        features, slices[index] + position * width + begin, mul
                    )
                    values.append(
                        program.scale(value, -1 if swapped and real == 1 else 1)
                    )
            value = program.concatenate(values)
            value = program.scale(
                value, math.sqrt((2 * min(ell, wigner_mmax) + 1) / len(rows))
            )
            matrix = program.gather(
                wigner,
                (offset + b * dim + a for a in range(dim) for b in rows)
                if packed
                else (
                    a * local_dim + b
                    for a in range(ell * ell, (ell + 1) ** 2)
                    for b in rows
                ),
            )
            outputs.append(program.matmul(matrix, value, dim, len(rows), mul))
        else:
            matrix = program.gather(
                wigner,
                (offset + a * dim + b for a in rows for b in range(dim))
                if packed
                else (
                    a * global_dim + b
                    for a in rows
                    for b in range(ell * ell, (ell + 1) ** 2)
                ),
            )
            value = program.matmul(
                matrix, program.slice(features, start, dim * mul), len(rows), dim, mul
            )
            for m, (index, begin) in enumerate(zip(indices, starts)):
                values = []
                for real in range(1 if m == 0 else 2):
                    swapped = basis_change and odd and m > 0
                    position = (1 - real) if swapped else real
                    a = 0 if m == 0 else 2 * m - 1 + position
                    values.append(
                        program.scale(
                            program.slice(value, a * mul, mul),
                            -1 if swapped and real == 0 else 1,
                        )
                    )
                parts[index].append((begin, mul, program.concatenate(values)))
    if inverse:
        return program.concatenate(outputs)
    outputs = []
    for (dim, _), parts_ir in zip(irreps, parts):
        parts_ir.sort()
        outputs.extend(
            program.slice(value, a * mul, mul)
            for a in range(dim)
            for _, mul, value in parts_ir
        )
    return program.concatenate(outputs)


def indexed_rotation(
    program, description, features, wigner, local_dim, global_dim,
    lmax, wigner_mmax, inverse, packed,
):
    """Encode a frame map as one multilinear contraction per edge tile."""
    mmax, basis_change, irreps, entries = description
    local_offsets, local_width = [], 0
    for dim, mul in irreps:
        local_offsets.append(local_width)
        local_width += dim * mul
    global_width = sum(mul * (2 * ell + 1) for _, mul, ell, *_ in entries)
    matrices, layout, offset = [], [], 0
    for ell in sorted({entry[2] for entry in entries}):
        dim, retained = 2 * ell + 1, min(ell, mmax)
        rows = (
            (ell, *(ell + s * m for m in range(1, retained + 1) for s in (1, -1)))
            if packed else wigner_indices(ell, retained, lmax)
        )
        packed_offset = ell * (4 * ell**2 - 1) // 3
        if packed:
            indices = (packed_offset + a * dim + b for a in rows for b in range(dim))
        elif inverse:
            indices = (
                b * local_dim + a for a in rows for b in range(ell**2, (ell + 1)**2)
            )
        else:
            indices = (
                a * global_dim + b for a in rows for b in range(ell**2, (ell + 1)**2)
            )
        matrix = program.gather(wigner, indices)
        if inverse:
            matrix = program.scale(
                matrix, math.sqrt((2 * min(ell, wigner_mmax) + 1) / len(rows))
            )
        matrices.append(matrix)
        for start, mul, degree, odd, local_indices, starts in entries:
            if degree != ell or not mul:
                continue
            local_rows = []
            for m, (index, begin) in enumerate(zip(local_indices, starts)):
                for real in range(1 if m == 0 else 2):
                    swapped = basis_change and odd and m > 0
                    row = 0 if m == 0 else 2 * m - 1 + (1 - real if swapped else real)
                    local_rows.append((
                        local_offsets[index] + real * irreps[index][1] + begin,
                        row, -1 if swapped and real == 0 else 1,
                    ))
            layout.append((start, mul, dim, offset, tuple(local_rows)))
        offset += len(rows) * dim
    role = 0 if inverse else 2
    operands = (
        program.constant(global_width, 0) if inverse else features,
        program.concatenate(matrices),
        features if inverse else program.constant(local_width, 0),
    )
    return program.add(
        "frame", global_width if inverse else local_width, operands,
        (repr((global_width, local_width, offset, tuple(layout))), role),
    )
