"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize(
    ("rotation_matrix", "axis"),
    [
        (o2.rotation_matrix_to_x_axis, 0),
        (o2.rotation_matrix_to_y_axis, 1),
        (o2.rotation_matrix_to_z_axis, 2),
    ],
)
def test_rotation_matrix_to_axis(rotation_matrix, axis):
    generator = torch.Generator().manual_seed(20260903)
    vectors = torch.randn(64, 3, dtype=DTYPE, generator=generator).to(DEVICE)
    rotation = rotation_matrix(vectors)
    rotated = torch.einsum("bij,bj->bi", rotation, vectors)
    expected = torch.zeros_like(vectors)
    expected[:, axis] = torch.linalg.vector_norm(vectors, dim=-1)

    torch.testing.assert_close(rotated, expected)
    identity = torch.eye(3, dtype=DTYPE, device=DEVICE).expand_as(rotation)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity)
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones(vectors.size(0), dtype=DTYPE, device=DEVICE),
    )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_alignment_backends(axis, backend, monkeypatch):
    from equivariantx.o2.rotation_matrix import alignment_quaternion
    from equivariantx.ops import config
    from equivariantx.ops.torch.o2.rotation_matrix import reference

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_ALIGNMENT_ADJOINT", True, raising=False)
    device = "cuda" if backend == "cuda" else "cpu"
    torch.manual_seed(20261113)
    x = (
        torch.randn(2, 3, dtype=torch.float64, device=device) * 0.2 + 0.5
    ).requires_grad_()
    operation = (
        o2.rotation_matrix_to_x_axis,
        o2.rotation_matrix_to_y_axis,
        o2.rotation_matrix_to_z_axis,
    )[axis]

    def call(v):
        return operation(v, eps=2e-6, backend=backend)

    actual, expected = call(x), reference(x, axis, eps=2e-6)
    for order in range(4):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        if order < 3:
            probe = torch.randn_like(actual) / actual.numel() ** 0.5
            actual, expected = [
                torch.autograd.grad((v.sin() * probe).sum(), x, create_graph=True)[0]
                for v in (actual, expected)
            ]
    torch.testing.assert_close(
        torch.func.jvp(call, (x,), (x,)),
        torch.func.jvp(lambda v: reference(v, axis, eps=2e-6), (x,), (x,)),
    )
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        actual = torch.autograd.forward_ad.unpack_dual(call(dual))
        expected = torch.autograd.forward_ad.unpack_dual(
            reference(dual, axis, eps=2e-6)
        )
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.vmap(call)(x), call(x))
    assert call(x[:0]).shape == (0, 3, 3)
    poles = torch.cat((torch.eye(3), -torch.eye(3))).to(x)
    torch.testing.assert_close(call(poles), reference(poles, axis, eps=2e-6))
    if axis == 1:
        torch.testing.assert_close(
            alignment_quaternion(x, backend=backend),
            reference(x, axis, quaternion=True),
        )
