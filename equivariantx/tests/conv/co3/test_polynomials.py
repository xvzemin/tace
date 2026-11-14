"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3


@pytest.mark.parametrize("degree", range(7))
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
def test_cartesian_harmonic_polynomials(degree, normalization, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3
    from equivariantx.conv.co3.polynomials import polynomial_coefficients

    vectors = torch.randn(8, 3, device="cuda")
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    vectors[:4] = torch.cat((torch.eye(3, device="cuda"), vectors.new_zeros(1, 3)))
    values = []
    for polynomial in polynomial_coefficients(degree, normalization):
        values.append(
            sum(c * vectors.pow(vectors.new_tensor(p)).prod(-1) for p, c in polynomial)
        )
    actual = torch.stack(values, dim=-1)
    expected = co3.CartesianHarmonics(degree, False, normalization).cuda()(vectors)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
def test_cartesian_harmonic_tables(normalization):
    from equivariantx.ops.common.cuda.o3.harmonics import polynomial_coefficients

    vectors = torch.randn(7, 3, dtype=torch.float64)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    for degree in range(9):
        actual = torch.stack(
            [
                sum(
                    coefficient * (vectors ** vectors.new_tensor(powers)).prod(-1)
                    for powers, coefficient in terms
                )
                for terms in polynomial_coefficients(degree, normalization)
            ],
            dim=-1,
        )
        expected = o3.spherical_harmonics(
            degree, vectors, normalize=False, normalization=normalization
        )
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
