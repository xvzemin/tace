"""Sparse delta/epsilon coefficients against direct tensor contractions."""

import pytest
import torch

from equivariantx.ops import get_kernel
from equivariantx.ops.common.torch.cartesian import normalized_coefficients


@pytest.mark.parametrize(
    "degrees",
    [(0, 3, 3), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 2, 1), (2, 3, 3), (3, 3, 4)],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_coefficients(degrees, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    l1, l2, l3 = degrees
    values = (
        torch.randn(2, 1, 3**l1, device=device, requires_grad=True),
        torch.randn(1, 3, 3**l2, device=device, requires_grad=True),
    )
    cg = torch.zeros(3**l1, 3**l2, 3**l3, device=device)
    for a, b, c, coefficient in normalized_coefficients(*degrees):
        cg[a, b, c] += coefficient
    expected = torch.einsum("...a,...b,abc->...c", *values, cg)
    for backend in ("torch", "cuda") if device == "cuda" else ("torch",):
        actual = get_kernel("cartesian_product", backend)(*values, *degrees)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        ga, gb = [
            torch.autograd.grad(result.sin().sum(), values, retain_graph=True)
            for result in (actual, expected)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-11, rtol=2e-11)
