"""Quaternion polynomials against independent Wigner degree recursion."""

import pytest
import torch

from equivariantx.o2 import WignerD
from equivariantx.ops.kernels.torch.quaternion import polynomial
from equivariantx.ops.quaternion import quaternion_wigner


@pytest.mark.parametrize("degree", [0, 1, 3, 6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_frames_and_derivatives(degree, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    axes = torch.eye(3, device=device, dtype=torch.float64)
    vectors = torch.cat((axes, -axes, axes + 0.03)).requires_grad_()
    frame = WignerD(degree, degree, method="recursive").to(device)
    actual = quaternion_wigner(vectors, degree)
    expected = frame.forward_packed(vectors)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    if degree:
        for _ in range(4):
            seed = torch.randn_like(actual) / actual.numel()
            actual, expected = [
                torch.autograd.grad((value * seed).sum(), vectors, create_graph=True)[0]
                for value in (actual, expected)
            ]
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        quaternion_wigner(vectors[:0], degree), frame.forward_packed(vectors[:0])
    )


def test_polynomial_zero_components_and_transforms():
    from equivariantx.ops.kernels.torch.quaternion import polynomial_indices

    polynomial_indices.cache_clear()
    q = torch.eye(4, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn_like(q)
    function = lambda value: polynomial(value, 3)
    for _ in range(2):
        _, jvp = torch.func.jvp(function, (q,), (tangent,))
    finite_difference = (
        function(q + 1e-5 * tangent) - function(q - 1e-5 * tangent)
    ) / 2e-5
    torch.testing.assert_close(jvp, finite_difference, atol=1e-7, rtol=1e-7)
    batch = torch.stack((q, q * 0.8))
    torch.testing.assert_close(
        torch.vmap(function)(batch), torch.stack([function(value) for value in batch])
    )
    assert torch.autograd.gradgradcheck(function, (q,), fast_mode=True)


def test_quaternion_compile():
    frame = WignerD(2, 3, method="quaternion")
    compiled = torch.compile(
        frame.forward_packed, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for rows in (7, 11):
        vectors = torch.randn(rows, 3, dtype=torch.float64, requires_grad=True)
        actual, expected = compiled(vectors), frame.forward_packed(vectors)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), vectors)[0],
            torch.autograd.grad(expected.square().sum(), vectors)[0],
        )


@pytest.mark.parametrize("rank", [0, 1, 2, 5])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("nodes", [0, 3])
def test_directional_polynomial(rank, transpose, nodes, double_precision):
    from equivariantx.ops.kernels.torch.quaternion import (
        polynomial,
        quaternion_polynomial,
    )

    lmax = 2
    q = torch.randn(nodes, 4) * 0.3
    q[:, :1] = 0
    directions = [torch.randn_like(q) * 0.2 for _ in range(rank)]
    width = sum((2 * degree + 1) ** 2 for degree in range(lmax + 1))
    cotangent = torch.randn(nodes, width)
    values = [q, *directions, *([cotangent] if transpose else [])]

    def directional(q, order):
        if order == 0:
            return polynomial(q, lmax)
        return torch.func.jvp(
            lambda x: directional(x, order - 1), (q,), (directions[order - 1],)
        )[1]

    expected = (
        torch.func.grad(lambda x: (directional(x, rank) * cotangent).sum())(q)
        if transpose
        else directional(q, rank)
    )
    torch.testing.assert_close(
        quaternion_polynomial(lmax, transpose, values), expected, atol=1e-10, rtol=1e-10
    )
