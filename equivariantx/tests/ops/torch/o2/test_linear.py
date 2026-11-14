"""Grouped channel paths compared with independent ordinary Torch paths."""

import pytest
import torch

from equivariantx import o2
from equivariantx.ops import config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_reference(monkeypatch, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = o2.Linear(
        "2x0e+3x0e+2x1m", "4x0e+2x1m+3x0e", biases=True, backend="torch"
    ).to(device=device, dtype=dtype)
    value = torch.randn(
        7, layer.irreps_in.dim, dtype=dtype, device=device, requires_grad=True
    )
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = layer(value)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = layer(value)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    losses = [x.sin().sum() for x in (actual, expected)]
    inputs = value, layer.weight, layer.bias
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=tolerance, rtol=tolerance)
        tangents = tuple(torch.randn_like(x) * 0.2 for x in inputs)
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
