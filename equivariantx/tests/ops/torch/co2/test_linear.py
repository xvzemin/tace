"""Cartesian channel paths against the ordinary Torch reference switch."""

import torch

from equivariantx import co2
from equivariantx.ops import config


def test_reference_paths(monkeypatch):
    layer = co2.Linear("2x0e+3x0e+2x2m", "4x0e+2x2m+3x0e", biases=True).double()
    value = torch.randn(7, layer.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = layer(value)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = layer(value)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    for x in (actual, expected):
        assert (
            torch.autograd.grad(x.sin().sum(), value, create_graph=True)[0]
            .isfinite()
            .all()
        )
