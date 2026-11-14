"""Cartesian representation metadata and basis compatibility."""

import pytest
import torch
from e3nn import o3

from equivariantx import co3

from ..utils.testing import (
    check_cartesian_representation_precision,
    check_cartesian_representation_transforms,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_representation_precision(device):
    check_cartesian_representation_precision(co3, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_representation_transforms(device, double_precision):
    check_cartesian_representation_transforms(co3, device)


@pytest.mark.parametrize("method", ["angles", "quaternion", "axis_angle"])
def test_rotation_input_precision(method):
    irrep = co3.Irrep("2e")
    low = torch.tensor(0.3, dtype=torch.bfloat16)
    high = torch.tensor(0.2, dtype=torch.float64)
    arguments = {
        "angles": (high, low, high),
        "quaternion": (torch.ones(4, dtype=torch.bfloat16),),
        "axis_angle": (torch.ones(3, dtype=torch.float64), low),
    }
    with pytest.raises(TypeError, match="float32 or float64"):
        getattr(irrep, "D_from_" + method)(*arguments[method])


def test_irreps_and_basis(double_precision):
    irreps = co3.Irreps("2x0e+1x1o+2x2e+1x2o+1x1o")
    assert irreps.dim == 2 + 3 + 27 + 3
    assert irreps.regroup() == co3.Irreps("2x0e+2x1o+1x2o+2x2e")
    assert co3.Irreps([co3.Irrep("1o"), co3.Irrep("2e")]).dim == 12
    assert irreps.filter(lmax=1).dim == 8
    with pytest.raises(ValueError):
        irreps.index("0o")
    to_cart = co3.ChangeOfBasis(irreps).to(DEVICE)
    to_sph = co3.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    x = torch.randn(4, irreps.spherical().dim, device=DEVICE)
    torch.testing.assert_close(to_sph(to_cart(x)), x, atol=2e-14, rtol=2e-14)
    for sign in (1, -1):
        matrix = sign * o3.rand_matrix(device=DEVICE)
        expected = to_cart(
            x @ irreps.spherical().D_from_matrix(matrix.cpu()).to(DEVICE).T
        )
        actual = to_cart(x) @ irreps.D_from_matrix(matrix).T
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    raw = torch.randn(4, irreps.dim, device=DEVICE)
    project = co3.Projector(irreps).to(DEVICE)
    torch.testing.assert_close(project(raw), to_cart(to_sph(raw)))
    torch.testing.assert_close(project(project(raw)), project(raw))
    assert irreps.randn(0, -1, device=DEVICE).shape == (0, irreps.dim)
