"""Tests for the corresponding EQX source module."""

import math
import pytest
import torch
from e3nn import o3
from equivariantx import co2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
