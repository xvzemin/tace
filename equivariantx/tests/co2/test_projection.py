"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import co2
from equivariantx.co3.symmetric import SymmetricBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transverse_reference(tensor, direction, rank):
    """Independent transverse projection using the two helicity projectors."""
    if rank == 0:
        return tensor + 0 * direction[..., :1]
    x, y, z = direction.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).unflatten(
        -1, (3, 3)
    )
    plane = torch.eye(
        3, device=direction.device, dtype=direction.dtype
    ) - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    matrix = (plane + 1j * cross) * 0.5
    value = tensor.to(matrix.dtype)
    for axis in range(rank):
        value = torch.einsum(
            "...aib,...ji->...ajb",
            value.unflatten(-1, (3**axis, 3, 3 ** (rank - axis - 1))),
            matrix,
        ).flatten(-3)
    return 2 * value.real


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 6, 8, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transverse_projector(rank, dtype):
    basis = SymmetricBasis(rank).to(device=DEVICE, dtype=dtype)
    module = co2.TransverseProjector(rank).to(device=DEVICE, dtype=dtype)
    compact = torch.randn(3, basis.dim, device=DEVICE, dtype=dtype)
    tensor = basis.unpack(compact)
    vectors = torch.tensor(
        [[0, 1, 0], [0, -1, 0], [0.2, -0.7, 0.6]], device=DEVICE, dtype=dtype
    )
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    tol = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(basis.pack(tensor), compact, atol=tol, rtol=tol)
    torch.testing.assert_close(
        tensor.square().sum(-1), compact.square().sum(-1), atol=tol, rtol=tol
    )
    actual = module(tensor, direction)
    expected = transverse_reference(
        tensor.double(),
        direction.double() / direction.double().norm(dim=-1, keepdim=True),
        rank,
    ).to(dtype)
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(module(actual, direction), actual, atol=tol, rtol=tol)
    assert module(tensor[:0], direction[:0]).shape == (0, 3**rank)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_planar_detracer(rank, dtype):
    basis = SymmetricBasis(rank).to(device=DEVICE, dtype=dtype)
    projector = co2.PlaneProjector(rank)
    detracer = co2.PlanarDetracer(rank, dtype=dtype, device=DEVICE)
    direction = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 1e-9], [0.2, -0.7, 0.6]],
        dtype=dtype,
        device=DEVICE,
    )
    direction = direction / direction.norm(dim=-1, keepdim=True)
    plane = co2.plane_projector(direction)
    compact = torch.randn(3, basis.dim, 2, dtype=dtype, device=DEVICE).transpose(-1, -2)
    tensor = basis.unpack(compact)
    projected = projector(tensor, plane[:, None])
    matrix = detracer.matrix(plane)
    actual = detracer(projected, plane[:, None])
    reference_direction = direction.double()
    reference_direction = reference_direction / reference_direction.norm(
        dim=-1, keepdim=True
    )
    expected = transverse_reference(
        tensor.double(), reference_direction[:, None], rank
    ).to(dtype)
    tol = 4e-5 if dtype == torch.float32 else 3e-12
    for value, reference in (
        (actual, expected),
        (actual, torch.einsum("eij,ecj->eci", matrix, projected)),
        (detracer(actual, plane[:, None]), actual),
        (projector(actual, plane[:, None]), actual),
        (matrix, matrix.transpose(-1, -2)),
    ):
        torch.testing.assert_close(value, reference, atol=tol, rtol=tol)
    assert detracer(projected[:0], plane[:0, None]).shape == (0, 2, 3**rank)
    assert projector(tensor[:0], plane[:0, None]).shape == (0, 2, 3**rank)
    assert detracer.matrix(plane[:0]).shape == (0, 3**rank, 3**rank)
    torch.testing.assert_close(detracer.matrix(plane[0]), matrix[0], atol=tol, rtol=tol)
    if rank >= 2:
        trace = (
            actual.unflatten(-1, (3, 3, 3 ** (rank - 2)))
            .diagonal(dim1=-3, dim2=-2)
            .sum(-1)
        )
        torch.testing.assert_close(trace, torch.zeros_like(trace), atol=tol, rtol=0)
    if rank == 2:
        p = plane.flatten(-2)
        expected = torch.eye(9, dtype=dtype, device=DEVICE) - 0.5 * p.unsqueeze(
            -1
        ) * p.unsqueeze(-2)
        torch.testing.assert_close(matrix, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_plane_projector_arbitrary_tensor(rank, double_precision):
    projector = co2.PlaneProjector(rank)
    tensor = torch.randn(2, 3**rank, 4, device=DEVICE).transpose(-1, -2)
    direction = torch.randn(2, 3, device=DEVICE)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    plane = co2.plane_projector(direction)
    projected = projector(tensor, plane[:, None])
    torch.testing.assert_close(projector(projected, plane[:, None]), projected)
    for axis in range(rank):
        longitudinal = torch.einsum(
            "ei,ecbij->ecbj",
            direction,
            projected.unflatten(-1, (3**axis, 3, 3 ** (rank - axis - 1))),
        )
        torch.testing.assert_close(
            longitudinal, torch.zeros_like(longitudinal), atol=1e-14, rtol=0
        )


def test_planar_detracer_precision_and_derivatives(double_precision):
    rank = 3
    projector = co2.PlaneProjector(rank)
    module = co2.PlanarDetracer(rank, dtype=torch.float32, device=DEVICE).double()
    reference = co2.PlanarDetracer(rank, dtype=torch.float64, device=DEVICE)
    for (_, value), (_, expected) in zip(
        module.named_buffers(), reference.named_buffers()
    ):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    basis = SymmetricBasis(rank).to(DEVICE)

    def forward(h, vectors):
        direction = vectors / vectors.norm(dim=-1, keepdim=True)
        plane = co2.plane_projector(direction[:, None])
        return module(projector(basis.unpack(h), plane), plane)

    h = torch.randn(1, 1, basis.dim, device=DEVICE, requires_grad=True)
    vectors = torch.tensor([[0.7, -1.1, 1.3]], device=DEVICE, requires_grad=True)
    assert torch.autograd.gradcheck(forward, (h, vectors), fast_mode=True)
    assert torch.autograd.gradgradcheck(forward, (h, vectors), fast_mode=True)
    compiled = torch.compile(forward, backend="aot_eager", dynamic=True, fullgraph=True)
    for count in (2, 0, 1):
        h = torch.randn(count, 2, basis.dim, device=DEVICE, requires_grad=True)
        vectors = torch.randn(count, 3, device=DEVICE, requires_grad=True)
        actual, expected = compiled(h, vectors), forward(h, vectors)
        torch.testing.assert_close(actual, expected)
        for value, target in zip(
            torch.autograd.grad(actual.square().sum(), (h, vectors)),
            torch.autograd.grad(expected.square().sum(), (h, vectors)),
        ):
            torch.testing.assert_close(value, target)


@pytest.mark.parametrize("rank", [-1, 0.5])
def test_planar_projection_validation(rank):
    with pytest.raises(ValueError):
        co2.PlaneProjector(rank)
    with pytest.raises(ValueError):
        co2.PlanarDetracer(rank)
