"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx import co2
from equivariantx.co3.symmetric import SymmetricBasis
from equivariantx.tests.utils.testing import transverse_reference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("kind", ["project", "detrace"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_backend_high_derivatives(monkeypatch, rank, kind, backend, double_precision):
    from copy import deepcopy

    from equivariantx.ops import config

    monkeypatch.setattr(config, "TORCH_TENSOR_AXIS", True)
    monkeypatch.setattr(config, "TORCH_PLANAR_DETRACING", True)
    monkeypatch.setattr(config, "TORCH_PLANAR_CUBIC", True)
    cls = co2.PlaneProjector if kind == "project" else co2.PlanarDetracer
    layer = cls(rank, backend=backend).to(DEVICE)
    reference = deepcopy(layer)
    reference.backend = "torch"
    x = (torch.randn(2, 3, 3**rank, device=DEVICE) * 0.2).requires_grad_()
    plane = (torch.randn(2, 1, 3, 3, device=DEVICE) * 0.2).requires_grad_()
    values = x, plane
    actual = layer(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = reference(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    losses = [v.sin().sum() for v in (actual, expected)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-9, rtol=3e-9)
        tangents = tuple(torch.randn_like(v) * 0.1 for v in values)
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
    tangents = tuple(torch.randn_like(v) * 0.1 for v in values)
    actual = torch.func.jvp(layer, values, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = torch.func.jvp(reference, values, tangents)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(
        torch.vmap(layer)(x, plane), layer(x, plane), atol=3e-11, rtol=3e-11
    )
    assert layer(x[:0], plane[:0]).shape == (0, 3, 3**rank)


def test_detracer_retained_storage():
    for rank in (6, 8):
        module = co2.PlanarDetracer(rank, dtype=torch.float64)
        assert sum(value.numel() for value in module.buffers()) < 4 * 3**rank * rank**2
    legacy = co2.PlanarDetracer(2, dtype=torch.float64)
    legacy.identity = torch.eye(9, dtype=torch.float64)
    plane = torch.diag(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(
        legacy.matrix(plane),
        co2.PlanarDetracer(2, dtype=torch.float64).matrix(plane),
    )
