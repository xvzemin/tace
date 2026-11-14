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


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_exact_sparse_harmonics(monkeypatch, backend, normalize, normalization):
    from equivariantx.ops import config
    from equivariantx.ops.torch.co2.cartesian_harmonics import reference

    monkeypatch.setattr(config, "TORCH_PLANAR_HARMONICS", True)
    layer = co2.CartesianHarmonics(
        [0, 1, 3, 5, 2, 3], normalize, normalization, backend=backend
    ).to(DEVICE, torch.float64)
    x = (
        torch.randn(3, 2, device=DEVICE, dtype=torch.float64) * 0.1 + 0.5
    ).requires_grad_()
    ordinary = lambda x: reference(layer, x)
    actual, expected = layer(x), ordinary(x)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        seed = torch.randn_like(actual) * 0.1
        actual, expected = [
            torch.autograd.grad((y.sin() * seed).sum(), x, create_graph=True)[0]
            for y in (actual, expected)
        ]
    tangent = torch.randn_like(x)
    torch.testing.assert_close(
        torch.func.jvp(layer, (x,), (tangent,)),
        torch.func.jvp(ordinary, (x,), (tangent,)),
        atol=2e-11,
        rtol=2e-11,
    )
    torch.testing.assert_close(
        torch.vmap(layer)(x), ordinary(x), atol=2e-11, rtol=2e-11
    )
    assert layer(x[:0]).shape == (0, layer.irreps_out.dim)
    empty = co2.CartesianHarmonics([], backend=backend).to(x)
    assert empty(x).shape == (3, 0)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_compiled_and_legacy_state(backend):
    from equivariantx.ops.torch.co2.cartesian_harmonics import reference

    layer = co2.CartesianHarmonics([0, 1, 2, 3], True, backend=backend).to(
        DEVICE, torch.float64
    )
    run = torch.compile(layer, backend="aot_eager", fullgraph=True, dynamic=True)
    for rows in (3, 7):
        x = torch.randn(rows, 2, device=DEVICE, dtype=torch.float64, requires_grad=True)
        actual, expected = run(x), reference(layer, x)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), x)[0],
            torch.autograd.grad(expected.sin().sum(), x)[0],
            atol=2e-11,
            rtol=2e-11,
        )
    del layer.harmonic_indices, layer.harmonic_scales, layer.backend
    layer = layer.float().double()
    torch.testing.assert_close(layer(x), reference(layer, x), atol=0, rtol=0)
