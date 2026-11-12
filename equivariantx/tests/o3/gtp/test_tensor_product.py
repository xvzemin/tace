"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import o3 as eqx_o3

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.parametrize("normalization", ["none", "component", "norm"])
def test_gaunt_quadrature_matches_coefficients(normalization, double_precision):
    args = ("2x2o+0e+2x1o+0e", "1e+2o+0e", "2x0o+2x1e+2x2o+2x3e")
    kwargs = dict(irrep_normalization=normalization, shared_weights=False)
    grid = eqx_o3.GauntTensorProduct(*args, **kwargs)
    coefficients = eqx_o3.GauntTensorProduct(*args, method="coefficients", **kwargs)
    x = torch.randn(2, 1, grid.irreps_in1.dim, requires_grad=True)
    y = torch.randn(1, 3, grid.irreps_in2.dim, requires_grad=True)
    weight = torch.randn(2, 3, grid.weight_numel, requires_grad=True)
    actual, expected = grid(x, y, weight), coefficients(x, y, weight)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad(
                (value.sin() * seed).sum(), (x, y, weight), create_graph=True
            )
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in items]) for items in gradients
        ]
