"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import o2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution")
@pytest.mark.parametrize("lmax", [2, 7])
def test_generator_frame_derivatives(lmax):
    generator = torch.Generator().manual_seed(43)
    vectors = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    vectors[0] = torch.tensor([0.0, -1.0, 0.0])
    vectors = vectors.cuda().requires_grad_()
    seeds = [
        torch.randn(x, generator=generator, dtype=vectors.dtype).cuda()
        for x in [
            (5, (lmax + 1) * (2 * lmax + 1) * (2 * lmax + 3) // 3),
            (5, 3),
            (5, 3),
        ]
    ]
    frame = o2.WignerD(lmax, lmax, backend="cuda").to(vectors)

    def derivatives(method):
        y = frame.forward_packed(vectors, method=method)
        outputs = [y]
        for seed in seeds:
            (y,) = torch.autograd.grad((y * seed).sum(), vectors, create_graph=True)
            outputs.append(y)
        return outputs

    for a, b in zip(derivatives("generator"), derivatives("quaternion")):
        torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution")
def test_generator_large_degree():
    degree = 24
    values = torch.tensor([[0.35, -0.71, 0.12]], dtype=torch.float64)
    output = []
    for device, method in (("cpu", "recursive"), ("cuda", "generator")):
        x = values.to(device).requires_grad_()
        frame = o2.WignerD(
            degree, degree, backend=device if device == "cuda" else "torch"
        ).to(x)
        block = frame.forward_packed(x, method=method)[0, -((2 * degree + 1) ** 2) :]
        (dx,) = torch.autograd.grad(block[::73].sum(), x)
        output.append((block.detach().cpu(), dx.cpu()))
    for a, b in zip(*output):
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9)


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
