"""Quadrature transforms, polynomial equivariance, and autograd."""

import pytest
import torch
from e3nn import o3

from eqx.nn import PolynomialActivation
from eqx.o2 import Irrep, Irreps, O2Grid
from eqx.o3 import S2Grid


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=["lebedev", "gauss_legendre", "equiangular"])
def quadrature(request):
    return request.param


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize(
    "irreps",
    [
        "",
        "0e",
        "0o",
        "2x0o+3x0e",
        "2x0e+2x1m+2x3m",
        "2x1m+0o+3x2m+1m+0e+0o",
        "2x0e+2x0o+4x1m+4x2m+4x3m",
        "2x32m",
        "0o+2x32m",
    ],
)
def test_o2_grid_round_trip(irreps, normalization, device):
    grid = O2Grid(
        irreps, normalization=normalization, dtype=torch.float64, device=device
    )
    expected_sheets = 2 if any(ir.is_odd_scalar() for ir, _ in grid.irreps) else 1
    assert grid.num_sheets == expected_sheets
    assert grid.grid_shape == (grid.num_channels, expected_sheets, grid.resolution)
    for count in (0, 4):
        x = torch.randn(grid.dim, count, 3, dtype=torch.float64, device=device).movedim(
            0, -1
        )
        values = grid(x)
        assert values.shape == (count, 3, *grid.grid_shape)
        torch.testing.assert_close(grid.from_grid(values), x, atol=3e-12, rtol=3e-12)
        if normalization == "integral":
            integral = (values.square() * grid.weights).sum((-3, -2, -1)) * (
                2 * torch.pi
            )
            torch.testing.assert_close(
                integral, x.square().sum(-1), atol=3e-12, rtol=3e-12
            )
    x = torch.randn(grid.dim, dtype=torch.float64, device=device)
    torch.testing.assert_close(grid.from_grid(grid(x)), x, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("mmax", [0, 6, 32])
def test_o2_grid_fourier_transform(mmax, normalization, double_precision, device):
    channels = 3
    irreps = Irreps(
        [(Irrep(0, 1), channels)]
        + [(Irrep(m, 0), channels) for m in range(1, mmax + 1)]
    )
    grid = O2Grid(irreps, normalization=normalization, device=device)
    x = irreps.randn(2, -1, device=device)
    coefficients = x.unflatten(-1, (2 * mmax + 1, channels)).transpose(-1, -2)
    alpha = torch.arange(grid.resolution, device=device) * (
        2 * torch.pi / grid.resolution
    )
    basis = [torch.ones_like(alpha)]
    for m in range(1, mmax + 1):
        basis.extend(((m * alpha).cos(), (m * alpha).sin()))
    basis = torch.stack(basis, -1)
    scale = torch.ones(2 * mmax + 1, device=device)
    if normalization == "integral":
        scale[0] /= (2 * torch.pi) ** 0.5
        scale[1:] /= torch.pi**0.5
    else:
        scale /= (mmax + 1) ** 0.5
        if normalization == "norm":
            scale[1:] *= 2**0.5
    synthesis = basis * scale
    norm = torch.full_like(scale, 0.5)
    norm[0] = 1
    analysis = (basis / (norm * scale * grid.resolution)).T
    expected = (coefficients @ synthesis.T).unsqueeze(-2)
    torch.testing.assert_close(grid(x), expected, atol=3e-12, rtol=3e-12)
    values = torch.randn_like(expected)
    expected = (values.squeeze(-2) @ analysis.T).transpose(-1, -2).flatten(-2)
    torch.testing.assert_close(grid.from_grid(values), expected, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(grid.from_grid(grid(x)), x, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(grid.weights.sum(), grid.weights.new_tensor(1))


def test_o2_grid_two_coefficient_sets(double_precision, device):
    grid = O2Grid("0e+0o+2x1m+2x2m", device=device)
    x = torch.randn(4, grid.dim, device=device)
    even = torch.stack((x[:, 0], x[:, 2], x[:, 4], x[:, 6], x[:, 8]), -1)
    odd = torch.stack((x[:, 1], x[:, 5], -x[:, 3], x[:, 9], -x[:, 7]), -1)
    even, odd = even @ grid.synthesis.T, odd @ grid.synthesis.T
    expected = torch.stack((even + odd, even - odd), -2).unsqueeze(-3) / 2**0.5
    torch.testing.assert_close(grid(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("irreps", ["2x0e+3x1m", "0e+0o+2x1m+2x2m", "0o"])
def test_o2_grid_sample_permutations(irreps, double_precision, device):
    grid = O2Grid(irreps, device=device)
    x = grid.irreps.randn(3, -1, device=device)
    rotation = grid.irreps.D_from_angle(2 * torch.pi / grid.resolution, device=device)
    reflection = grid.irreps.D_from_angle(0.0, reflected=True, device=device)
    index = -torch.arange(grid.resolution, device=device) % grid.resolution
    expected = grid(x).index_select(-1, index).flip(-2)
    torch.testing.assert_close(grid(x @ reflection.T), expected, atol=2e-13, rtol=2e-13)
    expected = grid(x).roll(1, -1)
    torch.testing.assert_close(grid(x @ rotation.T), expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("irreps", ["2x0e+3x1m+2m", "2x1m+0o+3x2m+1m+0e+0o", "0o"])
def test_o2_grid_polynomial_equivariance(
    irreps, normalization, reflected, double_precision, device
):
    irreps = Irreps(irreps)
    activation = PolynomialActivation("silu", device=device)
    grid = O2Grid(
        irreps,
        (activation.degree + 1) * irreps.mmax + 1,
        normalization=normalization,
        device=device,
    )
    rotation = irreps.D_from_angle(0.741, reflected=reflected, device=device)
    x = (irreps.randn(3, -1, device=device) * 0.2).requires_grad_()
    actual = grid.from_grid(activation(grid(x @ rotation.T)))
    expected = grid.from_grid(activation(grid(x))) @ rotation.T
    for order in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        if order < 2:
            seed = torch.randn_like(actual)
            actual, expected = [
                torch.autograd.grad((y * seed).sum(), x, create_graph=True)[0]
                for y in (actual, expected)
            ]


@pytest.mark.parametrize("irreps", ["", "0e+1m", "0e+0o+2x1m"])
def test_o2_grid_dtype_and_derivatives(irreps, double_precision, device):
    grid = O2Grid(irreps, dtype=torch.float32).to(device, torch.float64)
    reference = O2Grid(irreps, dtype=torch.float64, device=device)
    for actual, expected in zip(grid.buffers(), reference.buffers()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not grid.state_dict()
    x = torch.randn(1, grid.dim, device=device, requires_grad=True)

    def nonlinear(x):
        return grid.from_grid(torch.nn.functional.silu(grid(x)))

    if grid.dim:
        assert torch.autograd.gradcheck(nonlinear, (x,))
        assert torch.autograd.gradgradcheck(nonlinear, (x,))
    value = nonlinear(x)
    for _ in range(3):
        value = torch.autograd.grad(value.sin().sum(), x, create_graph=True)[0]
        assert torch.isfinite(value).all()
    torch.testing.assert_close(torch.vmap(nonlinear)(x), nonlinear(x))
    tangent = torch.randn_like(x)
    _, actual = torch.func.jvp(nonlinear, (x,), (tangent,))
    _, expected = torch.autograd.functional.jvp(nonlinear, x, tangent)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("irreps", ["", "0e+2x1m", "0o+2x1m+0e+2m"])
def test_o2_grid_compile(irreps, double_precision, device):
    grid = O2Grid(irreps, device=device)

    def nonlinear(x):
        return grid.from_grid(torch.nn.functional.silu(grid(x)))

    compiled = torch.compile(
        nonlinear, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        x = torch.randn(count, 2, grid.dim, device=device, requires_grad=True)
        actual, expected = compiled(x), nonlinear(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual, expected = [
            torch.autograd.grad(y.square().sum(), x)[0] for y in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_o2_grid_invalid_inputs(device):
    for irreps in ("0eo", "0oo", "1mo"):
        with pytest.raises(ValueError, match="time-even"):
            O2Grid(irreps)
    with pytest.raises(ValueError, match="resolution"):
        O2Grid("2m", resolution=4)
    with pytest.raises(ValueError, match="normalization"):
        O2Grid("0e", normalization="invalid")
    grid = O2Grid("0e+0o+1m", device=device)
    with pytest.raises(ValueError, match="features"):
        grid(torch.zeros(grid.dim + 1, device=device))
    with pytest.raises(ValueError, match="grid dimensions"):
        grid.from_grid(
            torch.zeros(grid.num_channels, 1, grid.resolution, device=device)
        )


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
def test_o2_grid_polynomial_projection(power, reflection, double_precision, device):
    grid = O2Grid("0e+1m+2m+3m+4m+5m", (power + 1) * 5 + 1).to(device)
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


def test_s2_dtype_and_derivatives(double_precision, device):
    grid = S2Grid(1, dtype=torch.float32, quadrature="gauss_legendre").to(
        device, torch.float64
    )
    reference = S2Grid(
        1, dtype=torch.float64, device=device, quadrature="gauss_legendre"
    )
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


@pytest.mark.parametrize("cls", [O2Grid, S2Grid])
def test_nonlinear_convergence(cls, double_precision, device):
    kwargs = {} if cls is O2Grid else {"quadrature": "gauss_legendre"}
    irreps = "0e+1m+2m" if cls is O2Grid else 2
    grids = [cls(irreps, resolution, **kwargs).to(device) for resolution in (9, 25, 49)]
    x = torch.randn(2, grids[0].dim, device=device)
    outputs = [grid.from_grid(torch.nn.functional.silu(grid(x))) for grid in grids]
    coarse = (outputs[0] - outputs[2]).abs().max()
    fine = (outputs[1] - outputs[2]).abs().max()
    assert fine < coarse * 0.01


def test_s2_compile(double_precision, device):
    grid = S2Grid(2, device=device, quadrature="gauss_legendre")

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


def test_s2_invalid_inputs():
    with pytest.raises(ValueError):
        S2Grid(-1)
    with pytest.raises(ValueError):
        S2Grid(3, resolution=2)
    with pytest.raises(ValueError):
        S2Grid(2, normalization="invalid")
    grid = S2Grid(2)
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


@pytest.mark.parametrize(
    "representation", ["0e+1m+2m+3m", "0e+0o+2x1m+2x2m+2x3m", "s2"]
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
