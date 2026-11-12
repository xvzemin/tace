"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import co2, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("m", range(7))
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_cartesian_planar_projector(double_precision, m, backend):
    if backend == "cuda" and DEVICE.type != "cuda":
        pytest.skip("CUDA is unavailable")
    matrix = co2.projector_matrix(m).to(DEVICE)
    basis = co2.path_matrix(m).to(DEVICE)
    torch.testing.assert_close(matrix, basis @ basis.T, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(matrix @ matrix, matrix, atol=2e-15, rtol=2e-15)
    decomposition = o2.ICTD(m).to(DEVICE)
    index = next(i for i, path in enumerate(decomposition.paths) if path[-1].m == m)
    path = decomposition.path_matrix(index)
    torch.testing.assert_close(matrix, path @ path.T, atol=2e-14, rtol=2e-14)
    irrep = "0e" if m == 0 else f"{m}m"
    irreps = co2.Irreps(f"2x{irrep}+1x{irrep}")
    value = torch.randn(3, irreps.dim, device=DEVICE, requires_grad=True)
    actual = co2.Projector(irreps, backend=backend).to(DEVICE)(value)
    expected = co2.Projector(irreps, method="factorized").to(DEVICE)(value)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
        actual, expected = [
            torch.autograd.grad(x.sin().sum(), value, create_graph=True)[0]
            for x in (actual, expected)
        ]


@pytest.mark.parametrize("m", [0, 1, 2, 3, 8, 16])
def test_path_matrix(m):
    tensor = torch.ones(1, dtype=torch.complex128, device=DEVICE)
    vector = torch.tensor([1, 1j], dtype=torch.complex128, device=DEVICE)
    for _ in range(m):
        tensor = torch.kron(tensor, vector)
    expected = (
        torch.stack((tensor.real, tensor.imag), dim=-1) * 2.0 ** ((1 - m) / 2)
        if m
        else tensor.real[:, None]
    )
    assert torch.equal(co2.path_matrix(m).to(DEVICE), expected)


def test_basis_and_representations(double_precision):
    irreps = co2.Irreps("2x0e+1x0oo+3x1mo+2x3m+1x1mo")
    assert list(irreps)[0] == (2, co2.Irrep("0e"))
    assert irreps.count("1mo") == 4
    assert irreps.regroup().num_irreps == irreps.num_irreps
    assert irreps.filter(mmax=0).dim == 3
    assert co2.Irreps(irreps.circular()) == irreps
    for m in range(9):
        matrix = co2.path_matrix(m).to(DEVICE)
        torch.testing.assert_close(
            matrix.T @ matrix, torch.eye(matrix.shape[1], device=DEVICE)
        )
    embed = co2.ChangeOfBasis(irreps).to(DEVICE)
    extract = co2.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    compact = irreps.circular().randn(4, -1, device=DEVICE)
    torch.testing.assert_close(extract(embed(compact)), compact)
    for reflected in (False, True):
        for time_reversal in (False, True):
            angle = torch.rand(4, device=DEVICE)
            d = irreps.D_from_angle(angle, reflected, time_reversal)
            expected = torch.einsum(
                "bij,bj->bi",
                irreps.circular().D_from_angle(angle, reflected, time_reversal),
                compact,
            )
            actual = extract(torch.einsum("bij,bj->bi", d, embed(compact)))
            torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    x = irreps.randn(3, -1, device=DEVICE)
    torch.testing.assert_close(co2.Projector(irreps).to(DEVICE)(x), x)
    assert irreps.randn(0, -1, device=DEVICE).shape == (0, irreps.dim)
    assert irreps.randn(-1, 0, device=DEVICE).shape == (irreps.dim, 0)
