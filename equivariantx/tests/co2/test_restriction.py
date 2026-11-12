"""Tests for the corresponding EQX source module."""

import math
import pytest
import torch
from equivariantx import co2, co3

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
