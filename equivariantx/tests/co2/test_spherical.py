"""Tests for the corresponding EQX source module."""

import math

import pytest
import torch
from e3nn import o3

from equivariantx import co2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.co2 import spherical as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module = co2.SphericalCoupling(2, 1, 2, backend="cuda").to(device)
    reference = co2.SphericalCoupling(2, 1, 2).to(device)
    for count in (3, 0):
        x, vectors = (
            torch.randn(count, 5, device=device),
            torch.randn(count, 3, device=device),
        )
        with patch.object(kernel, "forward", wraps=kernel.forward) as selected:
            actual = module(x, vectors)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(
            actual, reference(x, vectors), atol=2e-11, rtol=2e-11
        )
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


def test_coupling_recurrence_phase(monkeypatch, double_precision):
    from equivariantx.co2.spherical import coupling_polynomial, coupling_recurrence
    from equivariantx.o3.rotation import so3_generators

    reference = o3.wigner_3j

    def coefficients(l1, l2, l3, **kwargs):
        phase = -1 if l2 % 3 == 1 else 1
        return phase * reference(l1, l2, l3, **kwargs)

    caches = coupling_polynomial, coupling_recurrence, so3_generators
    for cached in caches:
        cached.cache_clear()
    try:
        monkeypatch.setattr(o3, "wigner_3j", coefficients)
        features = torch.randn(5, 5, device=DEVICE, requires_grad=True)
        vectors = torch.randn(5, 3, device=DEVICE, requires_grad=True)
        for l in range(5):
            module = co2.SphericalCoupling(2, l, 2, method="recurrence").to(DEVICE)
            actual = module(features, vectors)
            expected = torch.einsum(
                "...a,...b,abc->...c",
                features,
                o3.spherical_harmonics(l, vectors, True, "component"),
                coefficients(2, l, 2, device=DEVICE),
            )
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
            if l == 0:
                continue
            for _ in range(2):
                actual, expected = [
                    torch.autograd.grad(value.sin().sum(), vectors, create_graph=True)[
                        0
                    ]
                    for value in (actual, expected)
                ]
                torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    finally:
        for cached in caches:
            cached.cache_clear()


@pytest.mark.parametrize(
    "degrees",
    [(0, 3, 3), (2, 1, 2), (2, 3, 3), (3, 4, 2), (6, 6, 6), (10, 5, 7), (16, 8, 16)],
)
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("method", ["chebyshev", "recurrence"])
def test_spherical_coupling(
    degrees, normalization, method, double_precision, wigner_3j
):
    l1, l2, l3 = degrees
    cg = wigner_3j(*degrees, device=DEVICE)
    module = co2.SphericalCoupling(*degrees, normalization, method=method).to(DEVICE)
    vectors = torch.cat((torch.eye(3), -torch.eye(3), torch.randn(3, 3))).to(DEVICE)
    vectors.requires_grad_()
    features = torch.randn(9, 2 * l1 + 1, device=DEVICE, requires_grad=True)
    actual = module(features, vectors)
    expected = torch.einsum(
        "...a,...b,abc->...c",
        features,
        o3.spherical_harmonics(
            l2, vectors, normalize=True, normalization=normalization
        ),
        cg,
    )
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    if max(l1, l3) <= 3:
        cartesian = co2.O3TensorProduct(
            [(1, (l1, 1))],
            [(1, (l2, (-1) ** l2))],
            [(1, (l3, (-1) ** l2))],
            [(0, 0, 0, "uvu", False)],
            irrep_normalization="none",
            path_normalization="none",
            normalization=normalization,
            input_basis="spherical",
            output_basis="spherical",
        ).to(DEVICE)
        torch.testing.assert_close(
            actual, cartesian(features, vectors), atol=2e-11, rtol=2e-11
        )
    actual, expected = actual.sin(), expected.sin()
    for _ in range(3):
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        value,
                        (features, vectors),
                        probe,
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    assert module(features[:0], vectors[:0]).shape == (0, 2 * l3 + 1)


@pytest.mark.parametrize("l1,l3", [(0, 7), (7, 0), (3, 5), (8, 6), (16, 16)])
def test_coupling_recurrence(l1, l3, double_precision, wigner_3j):
    from equivariantx.co2.spherical import coupling_recurrence
    from equivariantx.o3.rotation import so3_generators

    degree, delta = min(l1, l3), abs(l1 - l3)
    basis = [
        wigner_3j(l1, l, l3, device=DEVICE)[:, l, :].T * math.sqrt(2 * l + 1)
        for l in range(delta, l1 + l3 + 1)
    ]
    coefficients = coupling_recurrence(l1, l3)
    matrix = so3_generators(degree).to(DEVICE)[1] / math.sqrt(
        max(1, degree * (degree + 1))
    )
    for k, value in enumerate(basis):
        actual = value @ matrix if l1 < l3 else matrix @ value
        expected = torch.zeros_like(actual)
        if k < len(coefficients):
            expected += coefficients[k] * basis[k + 1]
        if k:
            expected -= coefficients[k - 1] * basis[k - 1]
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("degrees", [(5, 10, 5), (8, 12, 8), (16, 12, 16)])
def test_spherical_recurrence_float32(degrees, wigner_3j):
    l1, l2, l3 = degrees
    cg = wigner_3j(*degrees, dtype=torch.float64, device=DEVICE)
    features = torch.randn(16, 2 * l1 + 1, device=DEVICE, requires_grad=True)
    vectors = torch.randn(16, 3, device=DEVICE, requires_grad=True)
    module = co2.SphericalCoupling(*degrees, method="recurrence").to(DEVICE)
    actual = module(features, vectors)
    try:
        harmonics = o3.spherical_harmonics(l2, vectors.double(), True, "component")
    except NotImplementedError:
        # Use Wigner matrices above the reference implementation's degree limit.
        alpha, beta = o3.xyz_to_angles(vectors.double().cpu())
        harmonics = o3.wigner_D(l2, alpha, beta, torch.zeros_like(alpha))[..., l2]
        harmonics = harmonics.to(DEVICE) * math.sqrt(2 * l2 + 1)
    expected = torch.einsum(
        "...a,...b,abc->...c",
        features.double(),
        harmonics,
        cg,
    )
    for _ in range(3):
        error = (actual.double() - expected.double()).norm()
        assert error <= 2e-4 * expected.norm()
        actual, expected = [
            torch.cat(
                [
                    x.flatten()
                    for x in torch.autograd.grad(
                        value.sin().sum() / value.numel() ** 0.5,
                        (features, vectors),
                        create_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]


@pytest.mark.parametrize("backend", ["torch", "reference", "cuda"])
@pytest.mark.parametrize("method", ["chebyshev", "recurrence"])
@pytest.mark.parametrize("degrees", [(1, 1, 1), (2, 3, 3), (3, 4, 2)])
def test_sparse_coupling_derivatives(
    backend, method, degrees, monkeypatch, double_precision, device
):
    from equivariantx.ops import config
    from equivariantx.ops.torch.co2.spherical import reference

    monkeypatch.setattr(config, "TORCH_REFERENCE", backend == "reference")
    monkeypatch.setattr(config, "TORCH_SPHERICAL_COUPLING", True)
    monkeypatch.setattr(config, "TORCH_SPHERICAL_ADJOINT", True)
    layer = co2.SphericalCoupling(
        *degrees, method=method, backend="cuda" if backend == "cuda" else "torch"
    ).to(device)
    x = (torch.randn(2, 3, 2 * (2 * degrees[0] + 1), device=device) * 0.1)[
        ..., ::2
    ].requires_grad_()
    v = (torch.randn(2, 1, 3, device=device) + 0.5).requires_grad_()
    actual, expected = layer(x, v), reference(layer, x, v, method)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        actual, expected = [
            sum(
                g.square().sum()
                for g in torch.autograd.grad(y.sin().sum(), (x, v), create_graph=True)
            )
            * 0.01
            for y in (actual, expected)
        ]
    directions = tuple(torch.randn_like(t) for t in (x, v))
    torch.testing.assert_close(
        torch.func.jvp(layer, (x, v), directions),
        torch.func.jvp(lambda a, b: reference(layer, a, b, method), (x, v), directions),
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        torch.vmap(layer)(x, v), layer(x, v), atol=2e-10, rtol=2e-10
    )
    assert layer(x[:0], v[:0]).shape == (0, 3, 2 * degrees[2] + 1)


def test_sparse_coupling_mixed_modes_and_legacy(monkeypatch, double_precision):
    from equivariantx.ops import config
    from equivariantx.ops.torch.co2.spherical import reference

    monkeypatch.setattr(config, "TORCH_SPHERICAL_COUPLING", True)
    layer = co2.SphericalCoupling(2, 3, 2)
    x = (torch.randn(2, 5) * 0.1).requires_grad_()
    v = (torch.randn(2, 3) + 0.5).requires_grad_()

    def oracle(a, b):
        return reference(layer, a, b, layer.method)

    assert torch.autograd.gradcheck(layer, (x, v))
    assert torch.autograd.gradgradcheck(layer, (x, v))
    direction = (torch.randn_like(x), torch.randn_like(v))

    def fn(a, b):
        return layer(a, b).sin().sum()

    def ref(a, b):
        return oracle(a, b).sin().sum()

    torch.testing.assert_close(
        torch.func.jvp(torch.func.grad(fn, argnums=(0, 1)), (x, v), direction),
        torch.func.jvp(torch.func.grad(ref, argnums=(0, 1)), (x, v), direction),
        atol=2e-9,
        rtol=2e-9,
    )
    del layer.generator_indices
    torch.testing.assert_close(layer(x, v), oracle(x, v))
    layer.float().double()
    assert layer.generator_indices.dtype == torch.long
    torch.testing.assert_close(layer(x, v), oracle(x, v), atol=2e-11, rtol=2e-11)
    assert not layer.state_dict()


@pytest.mark.parametrize("degrees", [(2, 3, 2), (2, 3, 3), (3, 4, 2)])
@pytest.mark.parametrize("shape", [(3,), (2, 1, 3)])
def test_channel_shared_matrix(degrees, shape, monkeypatch, double_precision, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.co2.spherical import reference

    monkeypatch.setattr(config, "SPHERICAL_MATRIX_MIN_CHANNELS", 16)
    monkeypatch.setattr(config, "TORCH_SPHERICAL_COUPLING", False)
    layer = co2.SphericalCoupling(*degrees, method="recurrence").to(device)
    x = (torch.randn(2, 16, 2 * (2 * degrees[0] + 1), device=device) * 0.1)[
        ..., ::2
    ].requires_grad_()
    v = (torch.randn(*shape, device=device) + 0.5).requires_grad_()
    actual, expected = layer(x, v), reference(layer, x, v, layer.method)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        actual, expected = [
            sum(
                g.square().sum()
                for g in torch.autograd.grad(y.sin().sum(), (x, v), create_graph=True)
            )
            * 0.001
            for y in (actual, expected)
        ]
    tangents = tuple(torch.randn_like(t) for t in (x, v))
    torch.testing.assert_close(
        torch.func.jvp(layer, (x, v), tangents),
        torch.func.jvp(
            lambda a, b: reference(layer, a, b, layer.method), (x, v), tangents
        ),
        atol=2e-10,
        rtol=2e-10,
    )
    assert layer(x[:0], v if v.ndim == 1 else v[:0]).shape == (
        0,
        16,
        2 * degrees[2] + 1,
    )


@pytest.mark.parametrize("degrees", [(2, 3, 2), (2, 3, 3), (6, 6, 6)])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_shared_matrix_fp32(degrees, backend, device):
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        layer = co2.SphericalCoupling(*degrees, backend=backend).to(device)
        oracle = co2.SphericalCoupling(*degrees).to(device=device, dtype=torch.float64)
        x = torch.randn(3, 16, 2 * degrees[0] + 1, device=device) * 0.1
        v = torch.randn(3, 1, 3, device=device) + 0.5
        w = torch.randn(16, 1, device=device) * 0.1
        results = []
        for module, dtype in ((layer, torch.float32), (oracle, torch.float64)):
            a = x.to(dtype).detach().requires_grad_()
            b = v.to(dtype).detach().requires_grad_()
            c = w.to(dtype).detach().requires_grad_()
            y = module(a * c, b)
            energy = 1 + y.sin().sum() / y.numel() ** 0.5
            force = -torch.autograd.grad(energy, b, create_graph=True)[0]
            loss = (force + 0.01).square().sum() + energy.square()
            gradient = torch.autograd.grad(loss, c)[0]
            results.append((y, force, gradient))
        for actual, expected in zip(*results):
            error = (actual.double() - expected).norm() / expected.norm().clamp_min(
                1e-20
            )
            assert error < 5e-5
    finally:
        torch.set_float32_matmul_precision(precision)
