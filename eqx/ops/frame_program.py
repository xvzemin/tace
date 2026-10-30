"""Local-frame expressions."""

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
