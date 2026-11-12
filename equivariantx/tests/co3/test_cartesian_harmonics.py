"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("normalize", [False, True])
def test_harmonics(double_precision, normalization, normalize):
    irreps = co3.Irreps.spherical_harmonics(6)
    module = co3.CartesianHarmonics(irreps, normalize, normalization).to(DEVICE)
    inverse = co3.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    x = torch.randn(8, 3, device=DEVICE, requires_grad=True) * 0.5
    actual = inverse(module(x))
    expected = o3.spherical_harmonics(list(range(7)), x, normalize, normalization)
    torch.testing.assert_close(actual, expected, atol=5e-13, rtol=5e-13)
    for _ in range(2):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.autograd.grad((v * seed).sum(), x, create_graph=True)[0]
            for v in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    assert module(x[:0]).shape == (0, irreps.dim)
    if not normalize:
        zero = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
        result = module(zero)
        assert torch.isfinite(torch.autograd.grad(result.sum(), zero)[0]).all()
