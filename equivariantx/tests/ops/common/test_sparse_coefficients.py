"""Sparse coefficient packing agrees with dense maps in every transpose."""

import torch

from equivariantx.ops.common.sparse_coefficients import bilinear_tables
from equivariantx.ops.common.torch.sparse_bilinear import reference


def test_duplicate_and_empty_entries():
    dimensions = (3, 2, 4)
    entries = (
        (0, 1, 2, 0.5),
        (0, 1, 2, 0.25),
        (2, 0, 1, -1.0),
        (1, 0, 0, 2.0),
        (1, 0, 0, -2.0),
    )
    matrix = torch.zeros(dimensions, dtype=torch.float64)
    for i, j, k, coefficient in entries:
        matrix[i, j, k] += coefficient
    tables = bilinear_tables(dimensions, entries)
    for role in range(3):
        perm = [k for k in range(3) if k != role] + [role]
        a, b = [torch.randn(5, dimensions[k], dtype=torch.float64) for k in perm[:2]]
        expected = torch.einsum("ni,nj,ijk->nk", a, b, matrix.permute(perm))
        torch.testing.assert_close(reference(a, b, tables, role), expected)
    empty = bilinear_tables((2, 3, 0), ())
    assert empty[-1].shape == (0, 1)
