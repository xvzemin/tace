"""Sparse coefficient rows preserve bilinear products and transposed gradients."""

import pytest
import torch

from equivariantx.ops.common.sparse import bilinear_rows
from equivariantx.ops.common.torch.sparse_axis import evaluate


@pytest.mark.parametrize("zero", [False, True])
def test_bilinear_rows(zero):
    coefficient = torch.randn(4, 3, 5, dtype=torch.float64)
    coefficient[coefficient.abs() < 0.7] = 0
    coefficient[..., 0] = 0
    if zero:
        coefficient.zero_()
    indices, columns, weights = bilinear_rows(coefficient)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
    actual = evaluate(x, indices, y[..., columns] * weights)
    expected = torch.einsum("...a,...b,abc->...c", x, y, coefficient)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        actual, expected = [
            sum(
                g.square().sum()
                for g in torch.autograd.grad(v.sin().sum(), (x, y), create_graph=True)
            )
            * 0.01
            for v in (actual, expected)
        ]
