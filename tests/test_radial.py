"""Spherical Bessel values, derivatives and radial normalization."""

import math

import numpy as np
import pytest
import torch
from scipy.special import spherical_jn

from tace.models.radial import (
    RadialBasis,
    compute_jn_zeros,
    jnSphericalBesselBasis,
    spherical_bessel_jn,
)

pytestmark = pytest.mark.usefixtures("double_precision")


def reference_derivative(n, x, derivative):
    if derivative == 0:
        return spherical_jn(n, np.abs(x)) * np.where(x < 0, (-1) ** n, 1)
    if n == 0:
        return -reference_derivative(1, x, derivative - 1)
    return (
        n * reference_derivative(n - 1, x, derivative - 1)
        - (n + 1) * reference_derivative(n + 1, x, derivative - 1)
    ) / (2 * n + 1)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 9, 10, 16, 32, 64, 128])
def test_spherical_bessel_values_and_derivatives(n):
    limit = math.sqrt(8 * n + 12) if n else 1.0
    positive = torch.cat(
        (
            torch.logspace(-12, math.log10(max(100, 3 * n)), 180),
            torch.tensor([limit, max(n, 1)])[:, None]
            .mul(torch.tensor([1 - 1e-8, 1.0, 1 + 1e-8]))
            .flatten(),
            # Include zeros of j0, used to normalize the old downward recurrence.
            torch.arange(1, 10) * math.pi,
        )
    )
    x = torch.cat((-positive, torch.zeros(1), positive)).requires_grad_()
    value = spherical_bessel_jn(n, x)
    for derivative in range(4):
        expected = torch.from_numpy(
            reference_derivative(n, x.detach().numpy(), derivative)
        )
        torch.testing.assert_close(value, expected, atol=2e-13, rtol=2e-11)
        if derivative == 0:
            mask = (x.abs() <= max(n, 1)) & (expected.abs() > 1e-250)
            torch.testing.assert_close(value[mask], expected[mask], atol=0, rtol=2e-11)
        if derivative < 3:
            value = torch.autograd.grad(value.sum(), x, create_graph=True)[0]


@pytest.mark.parametrize("n", [0, 1, 5, 10, 64, 128])
def test_spherical_bessel_zeros_and_normalization(n):
    roots = compute_jn_zeros(n, 5)
    assert roots.shape == (5,)
    assert roots[0] > n
    assert np.all(np.diff(roots) > 3.0)
    np.testing.assert_allclose(spherical_jn(n, roots), 0.0, atol=3e-15)
    if n == 0:
        np.testing.assert_array_equal(roots, np.arange(1, 6) * math.pi)
    if n == 1:
        np.testing.assert_allclose(
            roots[:3], [4.493409457909064, 7.725251836937707, 10.904121659428899]
        )
    # Interlacing checks that the first roots were not skipped.
    next_roots = compute_jn_zeros(n + 1, 4)
    assert np.all(roots[:-1] < next_roots)
    assert np.all(next_roots < roots[1:])

    basis = jnSphericalBesselBasis(cutoff=5.0, order=n, num_basis=5)
    points, weights = np.polynomial.legendre.leggauss(256)
    r = torch.tensor((points + 1) * 2.5).unsqueeze(-1)
    values = basis(r, None, None)
    integral = values.T @ (values * r.square() * torch.tensor(weights * 2.5)[:, None])
    torch.testing.assert_close(integral, torch.eye(5), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        basis(torch.tensor([[5.0]]), None, None), torch.zeros(1, 5), atol=2e-13, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("trainable", [False, True])
def test_spherical_bessel_basis_layout_and_state(dtype, trainable):
    basis = jnSphericalBesselBasis(
        order=[2, 0, 10], num_basis=[3, 2, 1], trainable=trainable
    ).to(dtype)
    r = torch.tensor([[0.0], [1e-10], [0.2], [3.0]], dtype=dtype, requires_grad=True)
    value = basis(r, None, None)
    argument = (
        basis.zeros.double().numpy(force=True)
        * r.double().numpy(force=True)
        / basis.cutoff.item()
    )
    expected = np.concatenate(
        [
            spherical_jn(n, x)
            for n, x in zip(basis.order, np.split(argument, [3, 5], axis=-1))
        ],
        axis=-1,
    ) * basis.normalizer.double().numpy(force=True)
    torch.testing.assert_close(value, torch.tensor(expected, dtype=dtype))
    assert value.shape == (4, 6) and value.dtype == dtype
    assert basis(torch.empty(0, 1, dtype=dtype), None, None).shape == (0, 6)
    assert set(basis.state_dict()) == {"zeros", "normalizer", "cutoff"}
    clone = jnSphericalBesselBasis(
        order=[2, 0, 10], num_basis=[3, 2, 1], trainable=trainable
    ).to(dtype)
    clone.load_state_dict(basis.state_dict(), strict=True)
    torch.testing.assert_close(value, clone(r, None, None), atol=0, rtol=0)
    first = torch.autograd.grad(value.sum(), r, create_graph=True)[0]
    parameters = (r, basis.zeros) if trainable else (r,)
    for derivative in torch.autograd.grad(first.square().sum(), parameters):
        assert torch.isfinite(derivative).all()


def test_spherical_bessel_trainable_gradients():
    basis = jnSphericalBesselBasis(
        order=[0, 1, 12], num_basis=[1, 1, 1], trainable=True
    )
    r = torch.tensor([[0.0], [0.2], [3.7]], requires_grad=True)

    def evaluate(r, zeros):
        return torch.func.functional_call(basis, {"zeros": zeros}, (r, None, None))

    assert torch.autograd.gradcheck(evaluate, (r, basis.zeros))
    assert torch.autograd.gradgradcheck(evaluate, (r, basis.zeros))
    # Trainable frequencies may pass through zero or change sign.
    with torch.no_grad():
        basis.zeros.copy_(torch.tensor([[0.0, -2.0, -12.0]]))
    assert torch.autograd.gradcheck(evaluate, (r, basis.zeros))
    assert torch.autograd.gradgradcheck(evaluate, (r, basis.zeros))


def test_spherical_bessel_compilation():
    basis = RadialBasis(radial_basis="jn", order=[0, 3, 12], num_basis=[2, 1, 1])
    compiled = torch.compile(basis, backend="aot_eager", fullgraph=True, dynamic=True)
    for size in (4, 7, 0):
        r = torch.linspace(0, 5.9, size).unsqueeze(-1).requires_grad_()
        args = (r, torch.empty(0), torch.empty(2, 0, dtype=torch.long), torch.empty(0))
        actual, _ = compiled(*args)
        expected, _ = basis(*args)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), r)[0]
        expected_grad = torch.autograd.grad(expected.sum(), r)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": -1},
        {"order": []},
        {"num_basis": 0},
        {"num_basis": []},
        {"order": [0, 1], "num_basis": [3]},
        {"cutoff": 0.0},
        {"cutoff": float("inf")},
    ],
)
def test_spherical_bessel_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        jnSphericalBesselBasis(**kwargs)
