"""Local matrix arithmetic without reduced-precision products."""

import pytest
import torch

from equivariantx.ops.common.torch.matmul import matmul, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("inner", [0, 7, 33])
def test_matmul(dtype, inner):
    left = torch.randn(2, 1, 3, inner, dtype=dtype, requires_grad=True)
    right = torch.randn(1, 4, inner, 5, dtype=dtype, requires_grad=True)
    expected = reference(left.double(), right.double())
    actual = matmul(left, right)
    tolerance = 3e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        actual.double(), expected, atol=tolerance, rtol=tolerance
    )
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value.sin().sum(), (left, right), create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=tolerance * 10, rtol=tolerance * 10)
        actual, expected = [
            sum(value.sum() for value in result) * 0.1 for result in gradients
        ]
