"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx.o3.s2grid import S2Grid

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.o3 import s2grid as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = S2Grid(2, quadrature="gauss_legendre", backend="cuda").to(device)
    reference = S2Grid(2, quadrature="gauss_legendre").to(device)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        x = torch.randn(count, module.dim, device=device)
        with patch.object(kernel, "to_grid", wraps=kernel.to_grid) as selected:
            grid = module(x)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(grid, reference(x))
        with patch.object(kernel, "from_grid", wraps=kernel.from_grid) as selected:
            actual = module.from_grid(grid.sin())
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(actual, reference.from_grid(grid.sin()))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = S2Grid(2, quadrature="gauss_legendre", device=device)
    features = torch.randn(3, layer.dim, device=device)
    check_method_precision(
        layer,
        [
            ("to_grid", (features,), {}),
            ("from_grid", (layer(features),), {}),
        ],
    )


@pytest.fixture(params=["lebedev", "gauss_legendre", "equiangular"])
def quadrature(request):
    return request.param


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


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("quadrature", ["lebedev", "gauss_legendre", "equiangular"])
def test_grid_backend(backend, quadrature, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o3 import s2grid as kernels

    monkeypatch.setattr(config, "TORCH_S2_GRID", True)
    layer = S2Grid(
        2, quadrature=quadrature, backend=backend, device=device, dtype=torch.float64
    )
    x = (
        torch.randn(2, layer.dim, device=device, dtype=torch.float64) * 0.1
    ).requires_grad_()
    actual = layer.from_grid(layer(x).sin())
    expected = kernels.from_grid_reference(
        layer, kernels.to_grid_reference(layer, x).sin()
    )
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        actual, expected = [
            torch.autograd.grad(v.sin().sum(), x, create_graph=True)[0]
            for v in (actual, expected)
        ]

    def run(value):
        return layer.from_grid(layer(value).sin())

    tangent = torch.randn_like(x)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, tangent)
        primal, actual_tangent = torch.autograd.forward_ad.unpack_dual(run(dual))
    expected_primal, expected_tangent = torch.autograd.functional.jvp(run, x, tangent)
    assert actual_tangent is not None
    torch.testing.assert_close(primal, expected_primal, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(actual_tangent, expected_tangent, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(
        torch.func.jvp(run, (x,), (tangent,)),
        torch.autograd.functional.jvp(run, x, tangent),
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(torch.vmap(run)(x), run(x))
    assert torch.autograd.gradcheck(run, (x,))
    assert torch.autograd.gradgradcheck(run, (x,))
    assert not layer.state_dict()
    empty = torch.empty(0, layer.dim, device=device, dtype=torch.float64)
    assert run(empty).shape == empty.shape


@pytest.mark.parametrize("lmax", [0, 6, 13])
@pytest.mark.parametrize("quadrature", ["gauss_legendre", "equiangular"])
def test_grid_high_order(lmax, quadrature, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o3 import s2grid as kernels

    monkeypatch.setattr(config, "TORCH_S2_GRID", True)
    layer = S2Grid(lmax, quadrature=quadrature, device=device, dtype=torch.float64)
    x = torch.randn(2, layer.dim, device=device, dtype=torch.float64)
    actual = layer(x)
    torch.testing.assert_close(
        actual, kernels.to_grid_reference(layer, x), atol=3e-12, rtol=3e-12
    )
    torch.testing.assert_close(layer.from_grid(actual), x, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("quadrature", ["lebedev", "gauss_legendre", "equiangular"])
def test_fp32_backend(backend, quadrature, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o3 import s2grid as kernels

    monkeypatch.setattr(config, "TORCH_S2_GRID", True)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        layer = S2Grid(
            3,
            quadrature=quadrature,
            backend=backend,
            dtype=torch.float32,
            device=device,
        )
        oracle = S2Grid(3, quadrature=quadrature, dtype=torch.float64, device=device)
        x = (
            torch.randn(2, layer.dim, device=device, dtype=torch.float32) * 0.1
        ).requires_grad_()
        precise = x.detach().double().requires_grad_()
        actual = layer.from_grid(layer(x).sin())
        expected = kernels.from_grid_reference(
            oracle, kernels.to_grid_reference(oracle, precise).sin()
        )
        for _ in range(3):
            torch.testing.assert_close(actual.double(), expected, atol=3e-7, rtol=1e-5)
            actual, expected = [
                torch.autograd.grad(value.sin().sum(), source, create_graph=True)[0]
                for value, source in ((actual, x), (expected, precise))
            ]
    finally:
        torch.set_float32_matmul_precision(precision)
