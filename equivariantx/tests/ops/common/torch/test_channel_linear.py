"""Shared linear values, recursive derivatives and functional transforms."""

from copy import deepcopy

import pytest
import torch

from equivariantx import co2, co3, o2
from equivariantx.ops import config


def linear_case(space, backend, device):
    namespace = {"o2": o2, "co2": co2, "co3": co3}[space]
    input_irreps = "2x0e+3x0e+2x1o+1x2e" if space == "co3" else "2x0e+3x0e+2x1m+1x2m"
    output_irreps = (
        "3x0e+2x1o+2x0e+2x2e+1x0o" if space == "co3" else "3x0e+2x1m+2x0e+2x2m+1x0o"
    )
    layer = namespace.Linear(
        input_irreps, output_irreps, biases=True, backend=backend
    ).to(device, torch.float64)
    inputs = (
        torch.randn(3, layer.irreps_in.dim, device=device, dtype=torch.float64) * 0.2,
        torch.randn(layer.weight_numel, device=device, dtype=torch.float64) * 0.2,
        torch.randn(layer.bias_numel, device=device, dtype=torch.float64) * 0.2,
    )
    return layer, tuple(value.requires_grad_() for value in inputs)


@pytest.mark.parametrize("space", ["o2", "co2", "co3"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_recursive_channel_maps(monkeypatch, space, backend, device):
    monkeypatch.setattr(config, "TORCH_CHANNEL_LINEAR", True)
    layer, inputs = linear_case(space, backend, device)
    reference = deepcopy(layer)
    reference.backend = "torch"
    actual = layer(*inputs)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = reference(*inputs)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    losses = [value.sin().sum() for value in (actual, expected)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
        tangents = tuple(torch.randn_like(value) * 0.1 for value in inputs)
        losses = [
            sum((g * t).sum() for g, t in zip(gradient, tangents))
            for gradient in gradients
        ]
    tangents = tuple(torch.randn_like(value) * 0.1 for value in inputs)
    actual = torch.func.jvp(layer, inputs, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = torch.func.jvp(reference, inputs, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    actual = torch.func.jvp(
        torch.func.grad(lambda x, w, b: layer(x, w, b).sin().sum(), argnums=(0, 1, 2)),
        inputs,
        tangents,
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = torch.func.jvp(
        torch.func.grad(
            lambda x, w, b: reference(x, w, b).sin().sum(), argnums=(0, 1, 2)
        ),
        inputs,
        tangents,
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
    x, w, b = inputs
    torch.testing.assert_close(
        torch.vmap(lambda row: layer(row, w, b))(x),
        layer(x, w, b),
        atol=3e-11,
        rtol=3e-11,
    )
    assert layer(x[:0], w, b).shape == (0, layer.irreps_out.dim)


@pytest.mark.parametrize("space", ["o2", "co2", "co3"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_dynamic_compile(monkeypatch, space, backend, device):
    monkeypatch.setattr(config, "TORCH_CHANNEL_LINEAR", True)
    layer, inputs = linear_case(space, backend, device)
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True, dynamic=True)
    for rows in (3, 7):
        x = torch.randn(
            rows,
            layer.irreps_in.dim,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        values = x, *inputs[1:]
        actual, expected = compiled(*values), layer(*values)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        gradients = [
            torch.autograd.grad(y.sin().sum(), values) for y in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
