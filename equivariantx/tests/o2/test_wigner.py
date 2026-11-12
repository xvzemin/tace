"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o2


def test_wigner_invalid_method(device):
    wigner = o2.WignerD(lmax=2, mmax=2).to(device)
    vectors = torch.randn(3, 3, device=device)
    with pytest.raises(ValueError, match="method"):
        wigner.matrix_blocks(vectors, method="invalid")


@pytest.mark.parametrize(("lmax", "mmax"), [(0, 0), (1, 1), (3, 1), (4, 2), (3, 3)])
def test_wigner_matches_o3_rotation_matrices(double_precision, lmax, mmax):
    vectors = torch.randn(3, 3, generator=torch.Generator().manual_seed(7))
    rotation = o2.rotation_matrix_to_y_axis(vectors)
    module = o2.WignerD(mmax, lmax)
    actual, inverse = module(vectors)
    full = torch.stack(
        [
            torch.block_diag(
                *[o3.Irrep(l, 1).D_from_matrix(r) for l in range(lmax + 1)]
            )
            for r in rotation
        ]
    )
    expected = full.index_select(1, module.local_indices)
    # The reference converts the rotation matrix through Euler angles.
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(
        inverse, expected.transpose(1, 2) * module.inverse_scale, atol=2e-9, rtol=2e-9
    )
    assert not module.state_dict()
    empty, empty_inverse = module(vectors[:0])
    assert empty.shape == (0, *actual.shape[1:])
    assert empty_inverse.shape == (0, *inverse.shape[1:])


@pytest.mark.parametrize("lmax", [2, 4])
def test_wigner_recursive_derivatives(double_precision, lmax):
    module = o2.WignerD(lmax, lmax, method="recursive")
    vectors = torch.randn(2, 3, requires_grad=True)
    assert torch.autograd.gradcheck(module.forward_packed, (vectors,), fast_mode=True)
    assert torch.autograd.gradgradcheck(
        module.forward_packed, (vectors,), fast_mode=True
    )
