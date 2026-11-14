"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
from equivariantx.tests.o2.testing import _transform
from equivariantx.tests.o2.testing import _time_reverse


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reflected", [False, True])
def test_circular_harmonics_is_equivariant(normalize, reflected):
    module = o2.CircularHarmonics(4, normalize=normalize).to(DEVICE, DTYPE)
    vectors = torch.randn(9, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    transformed = _transform(vectors, module.irreps_in, angle, reflected)
    expected = _transform(module(vectors), module.irreps_out, angle, reflected)
    torch.testing.assert_close(module(transformed), expected)


def test_time_odd_circular_harmonics_alternate_time_parity():
    harmonics = o2.CircularHarmonics(4, time_reversal=True).to(DEVICE, DTYPE)
    assert harmonics.irreps_in == o2.Irreps("1mo")
    assert harmonics.irreps_out == o2.Irreps("0ee+1mo+2me+3mo+4me")
    vectors = torch.randn(8, 2, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        harmonics(-vectors),
        _time_reverse(harmonics(vectors), harmonics.irreps_out),
    )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("mmax", [1, 4, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_recursive_derivatives(monkeypatch, backend, mmax, normalize):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o2.circular_harmonics import reference

    monkeypatch.setattr(config, "TORCH_CIRCULAR_ADJOINT", True)
    layer = o2.CircularHarmonics(mmax, normalize=normalize, backend=backend).to(DEVICE)
    value = (torch.randn(3, 2, device=DEVICE, dtype=DTYPE) * 0.1 + 0.5).requires_grad_()

    def ordinary(value):
        if normalize:
            value = torch.nn.functional.normalize(value, dim=-1)
        return reference(value, mmax)

    actual, expected = layer(value), ordinary(value)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        seed = torch.randn_like(actual) * 0.1
        actual, expected = [
            torch.autograd.grad((output.sin() * seed).sum(), value, create_graph=True)[
                0
            ]
            for output in (actual, expected)
        ]
    tangent = torch.randn_like(value)
    torch.testing.assert_close(
        torch.func.jvp(layer, (value,), (tangent,)),
        torch.func.jvp(ordinary, (value,), (tangent,)),
        atol=2e-11,
        rtol=2e-11,
    )
    torch.testing.assert_close(
        torch.vmap(layer)(value), ordinary(value), atol=2e-11, rtol=2e-11
    )
    actual = torch.func.jvp(
        torch.func.grad(lambda x: layer(x).sin().sum()), (value,), (tangent,)
    )
    expected = torch.func.jvp(
        torch.func.grad(lambda x: ordinary(x).sin().sum()), (value,), (tangent,)
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    assert layer(value[:0]).shape == (0, 1 + 2 * mmax)
