"""Tests for the corresponding EQX source module."""

import math
import pytest
import torch
from equivariantx import co2, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_harmonics(double_precision, normalization):
    module = co2.CartesianHarmonics(
        [0, 1, 3, 5, 2], normalization=normalization, time_reversal=True
    ).to(DEVICE)
    x = torch.randn(4, 2, device=DEVICE, requires_grad=True) * 0.2
    values = o2.circular_harmonics(x, 5, normalize=False)
    expected = []
    for m in module.orders:
        scale = 1 if normalization == "norm" or m == 0 else math.sqrt(2)
        if normalization == "integral":
            scale /= math.sqrt(2 * math.pi)
        expected.append(
            (values[..., :1] if m == 0 else values[..., 2 * m - 1 : 2 * m + 1]) * scale
        )
    actual = co2.ChangeOfBasis(module.irreps_out, True).to(DEVICE)(module(x))
    torch.testing.assert_close(actual, torch.cat(expected, -1))
    zero = torch.zeros(1, 2, device=DEVICE, requires_grad=True)
    grad = torch.autograd.grad(module(zero).sum(), zero, create_graph=True)[0]
    assert torch.isfinite(grad).all()
    assert torch.isfinite(torch.autograd.grad(grad.sum(), zero)[0]).all()
