"""Quadrature transforms, polynomial equivariance, and autograd."""

import pytest
import torch
from e3nn import o3

from eqx.nn import PolynomialActivation
from eqx.o2 import S1Grid
from eqx.o3 import S2Grid


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=["lebedev", "gauss_legendre", "equiangular"])
def quadrature(request):
    return request.param


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("mmax", [0, 6, 32])
def test_s1_round_trip(mmax, normalization, device):
    grid = S1Grid(mmax, normalization=normalization).to(device, torch.float64)
    for count in (0, 4):
        x = torch.randn(grid.dim, count, 3, dtype=torch.float64, device=device)
        x = x.movedim(0, -1)
        torch.testing.assert_close(grid.from_grid(grid(x)), x, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(grid.weights.sum(), grid.weights.new_tensor(1))


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("lmax", [0, 6, 13])
def test_s2_round_trip(lmax, normalization, quadrature, device):
    grid = S2Grid(lmax, normalization=normalization, quadrature=quadrature)
    grid = grid.to(device, torch.float64)
    for count in (0, 4):
        x = torch.randn(grid.dim, count, 3, dtype=torch.float64, device=device)
        x = x.movedim(0, -1)
        torch.testing.assert_close(grid.from_grid(grid(x)), x, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(grid.weights.sum(), grid.weights.new_tensor(1))


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_s2_e3nn(normalization, double_precision, device):
    grid = S2Grid(3, normalization=normalization, quadrature="equiangular").to(device)
    shape = (10, 10)
    synthesis = o3.ToS2Grid(3, shape, normalization=normalization).to(device)
    analysis = o3.FromS2Grid(shape, 3, normalization=normalization).to(device)
    x = torch.randn(2, 4, grid.dim, device=device, requires_grad=True)
    torch.testing.assert_close(
        grid(x), synthesis(x).flatten(-2), atol=1e-13, rtol=1e-13
    )
    actual = grid.from_grid(torch.nn.functional.silu(grid(x)))
    expected = analysis(torch.nn.functional.silu(synthesis(x)))
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.autograd.grad((value.sin() * seed).sum(), x, create_graph=True)[0]
            for value in (actual, expected)
        ]


@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("reflection", [False, True])
def test_s1_polynomial_equivariance(power, reflection, double_precision, device):
    grid = S1Grid(5, (power + 1) * 5 + 1).to(device)
    angle = torch.tensor(0.731, device=device)
    matrices = [torch.ones(1, 1, device=device)]
    for m in range(1, 6):
        c, s = (m * angle).cos(), (m * angle).sin()
        matrix = torch.stack((c, -s, s, c)).reshape(2, 2)
        if reflection:
            matrix = matrix @ torch.diag(angle.new_tensor([1, -1]))
        matrices.append(matrix)
    rotation = torch.block_diag(*matrices)
    x = torch.randn(3, 4, grid.dim, device=device)
    actual = grid.from_grid(grid(x @ rotation.T).pow(power))
    expected = grid.from_grid(grid(x).pow(power)) @ rotation.T
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("inversion", [False, True])
def test_s2_polynomial_equivariance(
    power, inversion, quadrature, double_precision, device
):
    grid = S2Grid(3, 3 * (power + 1), quadrature=quadrature).to(device)
    irreps = o3.Irreps.spherical_harmonics(3)
    rotation = o3.rand_matrix() * (-1 if inversion else 1)
    matrix = irreps.D_from_matrix(rotation).to(device)
    x = torch.randn(3, grid.dim, device=device)
    actual = grid.from_grid(grid(x @ matrix.T).pow(power))
    expected = grid.from_grid(grid(x).pow(power)) @ matrix.T
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("cls", [S1Grid, S2Grid])
def test_grid_dtype_and_derivatives(cls, double_precision, device):
    kwargs = {} if cls is S1Grid else {"quadrature": "gauss_legendre"}
    grid = cls(1, dtype=torch.float32, **kwargs).to(device, torch.float64)
    reference = cls(1, dtype=torch.float64, device=device, **kwargs)
    for actual, expected in zip(grid.buffers(), reference.buffers()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not grid.state_dict()
    x = torch.randn(1, grid.dim, device=device, requires_grad=True)

    def nonlinear(x):
        return grid.from_grid(torch.nn.functional.silu(grid(x)))

    assert torch.autograd.gradcheck(nonlinear, (x,))
    assert torch.autograd.gradgradcheck(nonlinear, (x,))
    value = nonlinear(x)
    for _ in range(3):
        value = torch.autograd.grad(value.sin().sum(), x, create_graph=True)[0]
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("cls", [S1Grid, S2Grid])
def test_nonlinear_convergence(cls, double_precision, device):
    kwargs = {} if cls is S1Grid else {"quadrature": "gauss_legendre"}
    grids = [cls(2, resolution, **kwargs).to(device) for resolution in (9, 25, 49)]
    x = torch.randn(2, grids[0].dim, device=device)
    outputs = [grid.from_grid(torch.nn.functional.silu(grid(x))) for grid in grids]
    coarse = (outputs[0] - outputs[2]).abs().max()
    fine = (outputs[1] - outputs[2]).abs().max()
    assert fine < coarse * 0.01


@pytest.mark.parametrize("cls", [S1Grid, S2Grid])
def test_grid_compile(cls, double_precision, device):
    kwargs = {} if cls is S1Grid else {"quadrature": "gauss_legendre"}
    grid = cls(2, device=device, **kwargs)

    def nonlinear(x):
        return grid.from_grid(torch.nn.functional.silu(grid(x)))

    compiled = torch.compile(
        nonlinear, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        x = torch.randn(count, 4, grid.dim, device=device, requires_grad=True)
        actual, expected = compiled(x), nonlinear(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=1e-12,
            rtol=1e-12,
        )


def test_extended_quadrature(device):
    for resolution, quadrature in [(131, "lebedev"), (140, "gauss_legendre")]:
        grid = S2Grid(2, resolution, quadrature=quadrature, dtype=torch.float64).to(
            device
        )
        assert grid.degree >= resolution
        for power in (2, 20, 130):
            integral = (grid.grid[:, 0].pow(power) * grid.weights).sum()
            torch.testing.assert_close(
                integral, integral.new_tensor(1 / (power + 1)), atol=1e-13, rtol=1e-13
            )
    with pytest.raises(ValueError, match="131"):
        S2Grid(2, 132)


@pytest.mark.parametrize("cls", [S1Grid, S2Grid])
def test_grid_invalid_inputs(cls):
    kwargs = {} if cls is S1Grid else {"quadrature": "gauss_legendre"}
    with pytest.raises(ValueError):
        cls(-1, **kwargs)
    with pytest.raises(ValueError):
        cls(3, resolution=2, **kwargs)
    with pytest.raises(ValueError):
        cls(2, normalization="invalid", **kwargs)
    grid = cls(2, **kwargs)
    with pytest.raises(ValueError):
        grid(torch.empty(2, grid.dim + 1))
    with pytest.raises(ValueError):
        grid.from_grid(torch.empty(2, 1))


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


@pytest.mark.parametrize("cls", [S1Grid, S2Grid])
@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "tanh", "sigmoid", "softplus"])
def test_polynomial_grid_equivariance(cls, act, double_precision, device):
    polynomial = PolynomialActivation(act).to(device)
    resolution = (polynomial.degree + 1) * 3
    if cls is S1Grid:
        from eqx.o2 import Irreps

        grid = cls(3, resolution + 1).to(device)
        irreps = Irreps("0e+1m+2m+3m")
        rotation = irreps.D_from_angle(
            0.741, reflected=True, dtype=torch.float64, device=device
        )
    else:
        grid = cls(3, resolution, quadrature="gauss_legendre").to(device)
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
