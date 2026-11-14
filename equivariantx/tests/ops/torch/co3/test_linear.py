"""Cartesian output grouping compared with separate per-path contractions."""

import pytest
import torch

from equivariantx import co3
from equivariantx.ops import config


@pytest.mark.parametrize("basis", ["cartesian", "spherical"])
def test_reference_paths(monkeypatch, basis):
    layer = co3.Linear(
        "2x0e+3x0e+2x2e", "4x0e+2x2e+3x0e", biases=True, output_basis=basis
    ).double()
    value = torch.randn(7, layer.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = layer(value)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = layer(value)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    losses = [x.sin().sum() for x in (actual, expected)]
    inputs = value, layer.weight, layer.bias
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-12, rtol=3e-12)
        tangents = tuple(torch.randn_like(x) * 0.2 for x in inputs)
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
