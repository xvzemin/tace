"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.nn.activation import PolynomialActivation
from equivariantx.o2.irreps import Irrep, Irreps
from equivariantx.o2.o2grid import O2Grid
from equivariantx.tests.o2.testing import _time_reverse, _transform


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.o2 import o2grid as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = O2Grid("2x0e+2x0o+4x1m", resolution=5, backend="cuda").to(device)
    reference = O2Grid("2x0e+2x0o+4x1m", resolution=5).to(device)
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
    layer = O2Grid("0e+0o+2x1m+2x2m", device=device)
    features = torch.randn(3, layer.dim, device=device)
    first = torch.randn(3, layer.irreps_in1.dim, device=device)
    check_method_precision(
        layer,
        [
            ("to_grid", (features,), {}),
            ("to_grid", (first, first.clone()), {}),
            ("from_grid", (layer(features),), {}),
        ],
    )


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize(
    "irreps",
    [
        "0e+0o",
        "2x0o+2x0e",
        "0e+0o+2x1m",
        "1m+0o+2m+1m+0e+2m",
        "2x0e+2x0o+4x1m+4x2m+4x3m",
        "0e+0o+" + "+".join(f"2x{m}m" for m in range(1, 33)),
    ],
)
def test_o2_grid_round_trip(irreps, normalization, device):
    grid = O2Grid(
        irreps, normalization=normalization, dtype=torch.float64, device=device
    )
    assert grid.num_sheets == 2
    assert grid.grid_shape == (grid.num_channels, 2, grid.resolution)
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
        [(Irrep(0, 1), channels), (Irrep(0, -1), channels)]
        + [(Irrep(m, 0), 2 * channels) for m in range(1, mmax + 1)]
    )
    grid = O2Grid(irreps, normalization=normalization, device=device)
    x1 = grid.irreps_in1.randn(2, -1, device=device)
    x2 = grid.irreps_in2.randn(2, -1, device=device)
    first = x1.unflatten(-1, (2 * mmax + 1, channels)).transpose(-1, -2)
    second = x2.unflatten(-1, (2 * mmax + 1, channels)).transpose(-1, -2)
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
    second_basis = [synthesis[:, 0]]
    for m in range(1, mmax + 1):
        second_basis.extend((-synthesis[:, 2 * m], synthesis[:, 2 * m - 1]))
    second_basis = torch.stack(second_basis, -1)
    even, odd = first @ synthesis.T, second @ second_basis.T
    expected = torch.stack((even + odd, even - odd), -2) / 2**0.5
    torch.testing.assert_close(grid(x1, x2), expected, atol=3e-12, rtol=3e-12)
    values = torch.randn_like(expected)
    positive, negative = values.unbind(-2)
    first = ((positive + negative) / 2**0.5) @ analysis.T
    second = ((positive - negative) / 2**0.5) @ analysis.T
    second = torch.stack(
        [second[..., 0]]
        + [
            value
            for m in range(1, mmax + 1)
            for value in (-second[..., 2 * m], second[..., 2 * m - 1])
        ],
        -1,
    )
    expected = tuple(value.transpose(-1, -2).flatten(-2) for value in (first, second))
    torch.testing.assert_close(
        grid.from_grid(values, split=True), expected, atol=3e-12, rtol=3e-12
    )
    torch.testing.assert_close(
        grid.from_grid(grid(x1, x2), split=True), (x1, x2), atol=3e-12, rtol=3e-12
    )
    torch.testing.assert_close(grid.weights.sum(), grid.weights.new_tensor(1))


def test_o2_grid_two_coefficient_sets(double_precision, device):
    grid = O2Grid("0e+0o+2x1m+2x2m", device=device)
    x = torch.randn(4, grid.dim, device=device)
    even = torch.stack((x[:, 0], x[:, 2], x[:, 4], x[:, 6], x[:, 8]), -1)
    odd = torch.stack((x[:, 1], x[:, 5], -x[:, 3], x[:, 9], -x[:, 7]), -1)
    even, odd = even @ grid.synthesis.T, odd @ grid.synthesis.T
    expected = torch.stack((even + odd, even - odd), -2).unsqueeze(-3) / 2**0.5
    torch.testing.assert_close(grid(x), expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("irreps", ["2x0e+2x0o+4x1m", "0e+0o+2x1m+2x2m", "0e+0o"])
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
@pytest.mark.parametrize(
    "irreps", ["2x0e+2x0o+4x1m+4x2m", "1m+0o+2m+1m+0e+2m", "0e+0o"]
)
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


@pytest.mark.parametrize("irreps", ["0e+0o", "0e+0o+2x1m", "1m+0o+2m+1m+0e+2m"])
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


@pytest.mark.parametrize("irreps", ["0e+0o", "0e+0o+2x1m", "1m+0o+2m+1m+0e+2m"])
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


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_o2_grid_time_reversal(normalization, device, double_precision):
    grid = O2Grid(
        "0oo+0ee+2x2mo+0eo+2x1me+2x1mo+0oe+2x2me",
        resolution=9,
        normalization=normalization,
    ).to(device)
    assert grid.num_sheets == 4
    x = grid.irreps.randn(4, -1, device=device).requires_grad_()
    values = grid(x)
    torch.testing.assert_close(grid.from_grid(values), x)
    first, second = grid.from_grid(values, split=True)
    torch.testing.assert_close(grid(first, second), values)
    torch.testing.assert_close(
        grid(_time_reverse(x, grid.irreps)),
        values.unflatten(-2, (2, 2)).flip(-3).flatten(-3, -2),
    )
    for reflected in (False, True):
        angle = x.new_tensor(0.37)
        actual = grid.from_grid(
            grid(_transform(x, grid.irreps, angle, reflected, True)).pow(3)
        )
        expected = _transform(
            grid.from_grid(values.pow(3)), grid.irreps, angle, reflected, True
        )
        for _ in range(3):
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
            probe = torch.randn_like(actual)
            actual, expected = [
                torch.autograd.grad(
                    (y * probe).sum(), x, create_graph=True, retain_graph=True
                )[0]
                for y in (actual, expected)
            ]


def test_o2_grid_invalid_inputs(device):
    for irreps in ("0eo", "0oo", "1mo"):
        with pytest.raises(ValueError, match="C copies"):
            O2Grid(irreps)
    with pytest.raises(ValueError, match="resolution"):
        O2Grid("0e+0o+2x1m+2x2m", resolution=4)
    with pytest.raises(ValueError, match="normalization"):
        O2Grid("0e+0o", normalization="invalid")
    for irreps in (
        "0ee+0oe+2x1me+0eo+2x1mo",
        "",
        "0e",
        "0o",
        "2x1m",
        "0e+0o+1m",
        "0e+2x0o+2x1m",
        "2x0e+2x0o+2x1m",
        "0e+0o+2x2m",
        "0e+0o+2x1m+4x2m",
    ):
        with pytest.raises(ValueError, match="C copies"):
            O2Grid(irreps)
    grid = O2Grid("0e+0o+2x1m", device=device)
    with pytest.raises(ValueError, match="features"):
        grid(torch.zeros(grid.dim + 1, device=device))
    with pytest.raises(ValueError, match="grid dimensions"):
        grid.from_grid(
            torch.zeros(grid.num_channels, 1, grid.resolution, device=device)
        )
    with pytest.raises(ValueError, match="coefficient set"):
        grid(torch.zeros(grid.dim, device=device), torch.zeros(grid.dim, device=device))
    with pytest.raises(ValueError, match="same shape"):
        grid(
            torch.zeros(2, grid.irreps_in1.dim, device=device),
            torch.zeros(1, grid.irreps_in2.dim, device=device),
        )


@pytest.mark.parametrize("mmax", [0, 1, 6])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_o2_grid_split_inputs(mmax, channels, normalization, double_precision, device):
    irreps = Irreps(
        [(Irrep(0, 1), channels), (Irrep(0, -1), channels)]
        + [(Irrep(m, 0), 2 * channels) for m in range(1, mmax + 1)]
    )
    grid = O2Grid(irreps, normalization=normalization, device=device)
    separate = O2Grid(
        grid.irreps_in1 + grid.irreps_in2, normalization=normalization, device=device
    )
    for count in (0, 4):
        first, second = [
            torch.randn(grid.irreps_in1.dim, count, 3, device=device)
            .movedim(0, -1)
            .requires_grad_()
            for _ in range(2)
        ]
        parts = [first[..., :channels], second[..., :channels]]
        for ir_slice in grid.irreps_in1.slices()[1:]:
            parts.append(
                torch.cat(
                    (
                        first[..., ir_slice].unflatten(-1, (2, channels)),
                        second[..., ir_slice].unflatten(-1, (2, channels)),
                    ),
                    -1,
                ).flatten(-2)
            )
        combined = torch.cat(parts, -1)
        actual = grid(first, second)
        expected = grid(combined)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            separate(torch.cat((first, second), -1)), expected, atol=0, rtol=0
        )
        torch.testing.assert_close(
            grid.from_grid(actual, split=True), (first, second), atol=3e-12, rtol=3e-12
        )
        torch.testing.assert_close(
            grid.from_grid(actual), combined, atol=3e-12, rtol=3e-12
        )
        for _ in range(2):
            actual, expected = [
                torch.cat(
                    torch.autograd.grad(
                        value.sin().sum(), (first, second), create_graph=True
                    ),
                    -1,
                )
                for value in (actual, expected)
            ]
            torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("mmax", [0, 2])
def test_o2_grid_split_derivatives_and_compile(mmax, double_precision, device):
    irreps = "0e+0o" + "".join(f"+2x{m}m" for m in range(1, mmax + 1))
    grid = O2Grid(irreps, device=device)

    def nonlinear(first, second):
        return grid.from_grid(torch.nn.functional.silu(grid(first, second)), split=True)

    first = grid.irreps_in1.randn(2, -1, device=device, requires_grad=True)
    second = grid.irreps_in2.randn(2, -1, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(nonlinear, (first, second))
    assert torch.autograd.gradgradcheck(nonlinear, (first, second))
    torch.testing.assert_close(
        torch.vmap(nonlinear)(first, second), nonlinear(first, second)
    )
    tangents = (torch.randn_like(first), torch.randn_like(second))
    _, actual = torch.func.jvp(nonlinear, (first, second), tangents)
    _, expected = torch.autograd.functional.jvp(nonlinear, (first, second), tangents)
    torch.testing.assert_close(actual, expected)
    compiled = torch.compile(
        nonlinear, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        first = grid.irreps_in1.randn(count, -1, device=device, requires_grad=True)
        second = grid.irreps_in2.randn(count, -1, device=device, requires_grad=True)
        actual, expected = compiled(first, second), nonlinear(first, second)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual, expected = [
            torch.autograd.grad(sum(x.square().sum() for x in result), (first, second))
            for result in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mmax", [1, 2, 6, 32])
def test_o2_grid_antisymmetric_product(mmax, double_precision, device):
    grid = O2Grid(
        "0e+0o" + "".join(f"+2x{m}m" for m in range(1, mmax + 1)), device=device
    )
    first = torch.zeros(8, grid.irreps_in1.dim, device=device)
    second = torch.zeros_like(first)
    a, b = torch.randn(2, 8, 2, device=device).unbind(0)
    first[..., -2:], second[..., -2:] = a, b
    _, odd = grid.from_grid(grid(first, second).square(), split=True)
    expected = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / (2 * (mmax + 1)) ** 0.5
    torch.testing.assert_close(odd[..., 0], expected, atol=2e-13, rtol=2e-13)
    _, swapped = grid.from_grid(grid(second, first).square(), split=True)
    torch.testing.assert_close(swapped[..., 0], -expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("reflection", [False, True])
def test_o2_grid_polynomial_projection(power, reflection, double_precision, device):
    grid = O2Grid("0e+0o+2x1m+2x2m+2x3m+2x4m+2x5m", (power + 1) * 5 + 1).to(device)
    rotation = grid.irreps.D_from_angle(0.731, reflected=reflection, device=device)
    x = torch.randn(3, 4, grid.dim, device=device)
    actual = grid.from_grid(grid(x @ rotation.T).pow(power))
    expected = grid.from_grid(grid(x).pow(power)) @ rotation.T
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("time_odd", [False, True])
def test_sheet_backend(backend, time_odd, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o2 import o2grid as kernels

    monkeypatch.setattr(config, "TORCH_O2_GRID", True)
    irreps = "0ee+0oe+2x1me+2x2me" + ("+0eo+0oo+2x1mo+2x2mo" if time_odd else "")
    layer = O2Grid(irreps, backend=backend, device=device, dtype=torch.float64)
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
    first, second = layer.from_grid(layer(x), split=True)
    torch.testing.assert_close(layer(first, second), layer(x), atol=3e-12, rtol=3e-12)
    assert not layer.state_dict()
    empty = torch.empty(0, layer.dim, device=device, dtype=torch.float64)
    assert run(empty).shape == empty.shape


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_fp32_backend(backend, monkeypatch, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.o2 import o2grid as kernels

    monkeypatch.setattr(config, "TORCH_O2_GRID", True)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        layer = O2Grid(
            "2x0e+2x0o+4x1m+4x2m", backend=backend, dtype=torch.float32, device=device
        )
        oracle = O2Grid("2x0e+2x0o+4x1m+4x2m", dtype=torch.float64, device=device)
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
