"""Sparse fixed-coefficient tables without dense Cartesian intermediates."""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def bilinear_coefficients(dimensions, entries):
    """Pack all three transposes of ``(left, right, output, coefficient)`` entries."""
    result = []
    for role in range(3):
        rows = [{} for _ in range(dimensions[role])]
        for *indices, coefficient in entries:
            other = tuple(indices[k] for k in range(3) if k != role)
            row = rows[indices[role]]
            row[other] = row.get(other, 0.0) + coefficient
        rows = [
            tuple((*pair, value) for pair, value in row.items() if value)
            for row in rows
        ]
        width = max(1, max((len(row) for row in rows), default=0))
        result.extend(
            tuple(
                tuple(row[k][i] if k < len(row) else 0 for k in range(width))
                for row in rows
            )
            for i in range(3)
        )
    return tuple(result)


def bilinear_tables(dimensions, entries, *, dtype=torch.float64, device="cpu"):
    """Create immutable gather indices and coefficients on the requested device."""
    return tuple(
        torch.tensor(
            data, dtype=dtype if i % 3 == 2 else torch.long, device=device
        ).reshape(dimensions[i // 3], -1)
        if dimensions[i // 3]
        else torch.empty(
            (0, 1), dtype=dtype if i % 3 == 2 else torch.long, device=device
        )
        for i, data in enumerate(bilinear_coefficients(dimensions, entries))
    )
