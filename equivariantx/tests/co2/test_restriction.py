"""Tests for the corresponding EQX source module."""

import math

import pytest
import torch

from equivariantx import co2, co3
from equivariantx.co2.restriction import transverse_quarter_turn
from equivariantx.co3.symmetric import SymmetricBasis
from equivariantx.tests.utils.testing import transverse_reference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.co2.restriction import TransverseProjector
    from equivariantx.ops import config
    from equivariantx.ops.cuda.co2 import restriction as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for cls, method in (
        (co2.Restriction, "restrict"),
        (TransverseProjector, "transverse"),
    ):
        module, reference = cls(3, backend="cuda").to(device), cls(3).to(device)
        for count in (3, 0):
            x = torch.randn(count, 7, device=device) @ co3.path_matrix(3).to(device).T
            direction = torch.nn.functional.normalize(
                torch.randn(count, 3, device=device), dim=-1
            )
            with patch.object(
                kernel, method, wraps=getattr(kernel, method)
            ) as selected:
                actual = module(x, direction)
            assert selected.call_count == (device == "cuda")
            torch.testing.assert_close(
                actual, reference(x, direction), atol=2e-11, rtol=2e-11
            )
        assert module.backend == "cuda"
    x, direction = torch.randn(3, 6, device=device), torch.randn(3, 3, device=device)
    with patch.object(kernel, "quarter_turn", wraps=kernel.quarter_turn) as selected:
        actual = transverse_quarter_turn(x, direction, backend="cuda")
    assert selected.call_count == (device == "cuda")
    torch.testing.assert_close(actual, transverse_quarter_turn(x, direction))
    assert config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = co2.Restriction(2).to(device)
    features = co3.Irreps("2e").randn(3, -1, device=device)
    direction = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(3, 3)
    values = layer(features, direction)
    check_method_precision(
        layer,
        [
            ("inverse", (values, direction), {}),
            ("inverse", (values, direction), {"project": False}),
        ],
    )


@pytest.mark.parametrize("l", range(7))
def test_restriction(double_precision, l):
    module = co2.Restriction(l).to(DEVICE)
    spherical = torch.randn(3, 2 * l + 1, device=DEVICE)
    x = spherical @ co3.path_matrix(l).to(DEVICE).T
    n = torch.randn(3, 3, device=DEVICE)
    n = n / n.norm(dim=-1, keepdim=True)
    values = module(x, n)
    torch.testing.assert_close(module.inverse(values, n), x, atol=8e-13, rtol=8e-13)
    raw = torch.zeros_like(x)
    for m, value in enumerate(values):
        for _ in range(l - m):
            value = (n.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
        raw = raw + co2.restriction_scale(l, m) * value
    torch.testing.assert_close(
        module.inverse(values, n, project=False), raw, atol=8e-13, rtol=8e-13
    )
    torch.testing.assert_close(
        sum(v.square().sum(-1) for v in values),
        spherical.square().sum(-1),
        atol=8e-13,
        rtol=8e-13,
    )
    for m, value in enumerate(values):
        if m:
            contraction = torch.einsum(
                "bik,bk->bi", value.reshape(3, 3 ** (m - 1), 3), n
            )
            torch.testing.assert_close(
                contraction, torch.zeros_like(contraction), atol=5e-13, rtol=0
            )
        if m > 1:
            trace = (
                value.reshape(3, 3 ** (m - 2), 3, 3).diagonal(dim1=-2, dim2=-1).sum(-1)
            )
            torch.testing.assert_close(
                trace, torch.zeros_like(trace), atol=5e-13, rtol=0
            )


@pytest.mark.parametrize("l", [1, 2, 4, 6, 10])
def test_restriction_derivatives(double_precision, l):
    module = co2.Restriction(l).float().double().to(DEVICE)
    h = torch.randn(2, 2, 2 * l + 1, device=DEVICE, requires_grad=True)
    vectors = torch.tensor(
        [[[0.0, -1.0, 0.0]], [[0.2, 0.7, -0.6]]], device=DEVICE, requires_grad=True
    )
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    tensor = h @ co3.path_matrix(l).to(DEVICE).T
    expected = []
    for m in range(l + 1):
        value = tensor
        for _ in range(l - m):
            value = (
                value.unflatten(-1, (value.shape[-1] // 3, 3)) * direction.unsqueeze(-2)
            ).sum(-1)
        expected.append(
            co2.restriction_scale(l, m) * transverse_reference(value, direction, m)
        )
    actual, expected = torch.cat(module(tensor, direction), -1), torch.cat(expected, -1)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    actual, expected = actual.sin(), expected.sin()
    for _ in range(3):
        probe = torch.randn_like(actual) / math.sqrt(actual.numel())
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        value, (h, vectors), probe, create_graph=True, retain_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)


def test_restriction_compile_and_broadcast(double_precision):
    for l in (0, 2, 4):
        module = co2.Restriction(l).to(DEVICE)
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        for count in (0, 3):
            h = torch.randn(count, 2, 2 * l + 1, device=DEVICE)
            tensor = h @ co3.path_matrix(l).to(DEVICE).T
            n = torch.tensor([[[0.0, 1.0, 0.0]]], device=DEVICE)
            values = compiled(tensor, n)
            torch.testing.assert_close(values, module(tensor, n))
            torch.testing.assert_close(
                module.inverse(values, n), tensor, atol=2e-12, rtol=2e-12
            )


@pytest.mark.parametrize("l", [4, 8, 10])
def test_restriction_single_precision(l):
    module = co2.Restriction(l).float().to(DEVICE)
    reference = co2.Restriction(l).double().to(DEVICE)
    h = torch.randn(3, 2 * l + 1, device=DEVICE)
    vectors = torch.tensor(
        [[0.0, -1.0, 0.0], [1e-6, 1.0, 1e-6], [0.2, 0.7, -0.6]], device=DEVICE
    )
    results = []
    for model, dtype in ((module, torch.float32), (reference, torch.float64)):
        r = vectors.to(dtype).detach().requires_grad_()
        tensor = h.to(dtype) @ co3.path_matrix(l).to(device=DEVICE, dtype=dtype).T
        value = torch.cat(model(tensor, r / r.norm(dim=-1, keepdim=True)), -1)
        grad = torch.autograd.grad(value[..., ::13].sin().sum(), r)[0]
        results.append((value, grad))
    torch.testing.assert_close(
        results[0][0], results[1][0].float(), atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        results[0][1], results[1][1].float(), atol=3e-4, rtol=3e-4
    )


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


@pytest.mark.parametrize("rank", [0, 1, 2, 4])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_backend_derivatives(rank, backend, monkeypatch, double_precision, device):
    from equivariantx.ops import config
    from equivariantx.ops.torch.co2 import restriction as reference

    monkeypatch.setattr(config, "TORCH_RESTRICTION", True)
    layer = co2.Restriction(rank, backend=backend).to(device)
    projector = co2.TransverseProjector(rank, backend=backend).to(device)
    x = (torch.randn(2, 3, 2 * 3**rank, device=device) * 0.1)[..., ::2].requires_grad_()
    v = (torch.randn(2, 1, 3, device=device) + 0.5).requires_grad_()

    def run(x, v, ordinary=False):
        n = torch.nn.functional.normalize(v, dim=-1)
        parts = reference.restrict_reference(layer, x, n) if ordinary else layer(x, n)
        value = (
            reference.transverse_reference(projector, x, n)
            if ordinary
            else projector(x, n)
        )
        tensors = tuple(p.sin() for p in parts)
        restored = (
            reference.inverse_reference(layer, tensors, n, True)
            if ordinary
            else layer.inverse(tensors, n)
        )
        result = [*parts, value, restored]
        if rank:
            turn = (
                reference.quarter_turn_reference
                if ordinary
                else lambda a, b: transverse_quarter_turn(a, b, backend=backend)
            )
            result.append(turn(parts[-1], n))
        return (
            torch.cat(torch.broadcast_tensors(*result), -1)
            if rank == 0
            else torch.cat(result, -1)
        )

    actual, expected = run(x, v), run(x, v, True)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        actual, expected = [
            sum(
                g.square().sum()
                for g in torch.autograd.grad(
                    y.sin().sum(), (x, v), create_graph=True, allow_unused=True
                )
                if g is not None
            )
            * 0.01
            for y in (actual, expected)
        ]
    directions = (torch.randn_like(x), torch.randn_like(v))
    torch.testing.assert_close(
        torch.func.jvp(lambda a, b: run(a, b), (x, v), directions),
        torch.func.jvp(lambda a, b: run(a, b, True), (x, v), directions),
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        torch.vmap(lambda a, b: run(a, b))(x, v), run(x, v), atol=2e-10, rtol=2e-10
    )
    assert run(x[:0], v[:0]).shape[0] == 0
