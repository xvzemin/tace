"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx.nn.activation import PolynomialActivation
from equivariantx.o2.irreps import Irreps
from equivariantx.o2.o2grid import O2Grid
from equivariantx.o3.s2grid import S2Grid


@pytest.mark.parametrize("cls", [O2Grid, S2Grid])
def test_nonlinear_convergence(cls, double_precision, device):
    kwargs = {} if cls is O2Grid else {"quadrature": "gauss_legendre"}
    irreps = "0e+0o+2x1m+2x2m" if cls is O2Grid else 2
    grids = [cls(irreps, resolution, **kwargs).to(device) for resolution in (9, 25, 49)]
    x = torch.randn(2, grids[0].dim, device=device)
    outputs = [grid.from_grid(torch.nn.functional.silu(grid(x))) for grid in grids]
    coarse = (outputs[0] - outputs[2]).abs().max()
    fine = (outputs[1] - outputs[2]).abs().max()
    assert fine < coarse * 0.01


@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "tanh", "sigmoid", "softplus"])
def test_polynomial_activation(act, double_precision, device):
    polynomial = PolynomialActivation(act, dtype=torch.float32).to(
        device, torch.float64
    )
    reference = PolynomialActivation(act, dtype=torch.float64, device=device)
    torch.testing.assert_close(
        polynomial.coefficients, reference.coefficients, atol=0, rtol=0
    )
    function = getattr(torch.nn.functional, act)
    x = torch.linspace(-3, 3, 1001, device=device)
    error = (polynomial(x) - function(x)).abs().max()
    assert error < 0.12
    refined = PolynomialActivation(act, degree=16, device=device)
    assert (refined(x) - function(x)).abs().max() < error
    if act == "tanh":
        torch.testing.assert_close(polynomial(-x), -polynomial(x), atol=0, rtol=0)
    elif act == "sigmoid":
        torch.testing.assert_close(
            polynomial(-x), 1 - polynomial(x), atol=1e-15, rtol=1e-15
        )
    else:
        torch.testing.assert_close(
            polynomial(x) - polynomial(-x), x, atol=2e-14, rtol=2e-14
        )
    x = torch.randn(4, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(polynomial, (x,))
    assert torch.autograd.gradgradcheck(polynomial, (x,))
    assert polynomial(x[:0]).shape == (0,)
    assert not polynomial.state_dict()


@pytest.mark.parametrize(
    "representation", ["1m+2m+3m+0e+1m+2m+3m+0o", "0e+0o+2x1m+2x2m+2x3m", "s2"]
)
@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "tanh", "sigmoid", "softplus"])
def test_polynomial_grid_equivariance(representation, act, double_precision, device):
    polynomial = PolynomialActivation(act).to(device)
    resolution = (polynomial.degree + 1) * 3
    if representation != "s2":
        irreps = Irreps(representation)
        grid = O2Grid(irreps, resolution + 1).to(device)
        rotation = irreps.D_from_angle(
            0.741, reflected=True, dtype=torch.float64, device=device
        )
    else:
        grid = S2Grid(3, resolution, quadrature="gauss_legendre").to(device)
        rotation = (
            o3.Irreps.spherical_harmonics(3).D_from_matrix(-o3.rand_matrix()).to(device)
        )
    x = torch.randn(2, 4, grid.dim, device=device, requires_grad=True)
    # Also test polynomial extrapolation, without clipping grid values.
    for scale in (1, 5):
        actual = grid.from_grid(polynomial(grid(scale * x @ rotation.T)))
        expected = grid.from_grid(polynomial(grid(scale * x))) @ rotation.T
        normalizer = expected.detach().abs().amax().clamp_min(1)
        torch.testing.assert_close(
            actual / normalizer, expected / normalizer, atol=2e-13, rtol=2e-12
        )
        seed = torch.randn_like(actual)
        for _ in range(2):
            actual, expected = [
                torch.autograd.grad((y * seed).sum(), x, create_graph=True)[0]
                for y in (actual, expected)
            ]
            normalizer = expected.detach().abs().amax().clamp_min(1)
            torch.testing.assert_close(
                actual / normalizer, expected / normalizer, atol=2e-13, rtol=2e-12
            )


def test_polynomial_compile_and_clenshaw(double_precision, device):
    polynomial = PolynomialActivation(degree=12).to(device)
    x = torch.linspace(-5, 5, 61, device=device, requires_grad=True)
    z = x / polynomial.bound
    previous, current = torch.ones_like(z), z
    expected = (
        polynomial.coefficients[0] * previous + polynomial.coefficients[1] * current
    )
    for i in range(2, polynomial.degree + 1):
        previous, current = current, 2 * z * current - previous
        expected = expected + polynomial.coefficients[i] * current
    torch.testing.assert_close(polynomial(x), expected, atol=2e-12, rtol=2e-12)
    compiled = torch.compile(
        polynomial, backend="aot_eager", fullgraph=True, dynamic=True
    )
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.isfinite(polynomial(torch.tensor([10.0], device=device))).all()
    assert not torch.allclose(
        polynomial(x.detach()), polynomial(x.detach().clamp(-3, 3))
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"act": "invalid"},
        {"degree": 0},
        {"degree": 1.5},
        {"bound": 0},
        {"bound": float("nan")},
    ],
)
def test_polynomial_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        PolynomialActivation(**kwargs)
