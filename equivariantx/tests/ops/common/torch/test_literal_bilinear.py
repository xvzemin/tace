"""Literal bilinear transposes agree with independent dense contractions."""

import pytest
import torch

from equivariantx.ops.common.torch.literal_bilinear import evaluate
from equivariantx.ops.common.torch.reduction import add_pairwise, sum_pairwise


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("dim", [-1, -2])
def test_literal_transposes(role, dim):
    dimensions = (2, 3, 2)
    entries = ((0, 0, 0, 0.5), (1, 2, 1, -0.75), (0, 1, 1, 0.25))
    matrix = torch.zeros(dimensions, dtype=torch.float64)
    for i, j, k, weight in entries:
        matrix[i, j, k] += weight
    others = [k for k in range(3) if k != role]
    x, y = [
        (torch.randn(2, 4, dimensions[k], dtype=torch.float64) * 0.1)
        .movedim(-1, dim)
        .requires_grad_()
        for k in others
    ]

    def reference(a, b):
        return torch.einsum(
            "...i,...j,ijk->...k",
            a.movedim(dim, -1),
            b.movedim(dim, -1),
            matrix.permute(*others, role),
        ).movedim(-1, dim)

    def call(a, b):
        return evaluate(a, b, dimensions, entries, role, dim)

    actual, expected = call(x, y), reference(x, y)
    for order in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        if order < 4:
            seed = torch.randn_like(actual) / actual.numel() ** 0.5
            actual, expected = [
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            (v.sin() * seed).sum(), (x, y), create_graph=True
                        )
                    ]
                )
                for v in (actual, expected)
            ]
    torch.testing.assert_close(
        torch.func.jvp(call, (x, y), (x, y)), torch.func.jvp(reference, (x, y), (x, y))
    )

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        got = torch.autograd.forward_ad.unpack_dual(call(dual, y))
        expected = torch.autograd.forward_ad.unpack_dual(reference(dual, y))
        torch.testing.assert_close(got, expected)


def test_empty_map():
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    for width in (0, 2):
        value = evaluate(x, y, (2, 3, width), ())
        torch.testing.assert_close(value, torch.zeros(3, width, dtype=x.dtype))
        gradients = torch.autograd.grad(value.square().sum(), (x, y), create_graph=True)
        for gradient, original in zip(gradients, (x, y)):
            torch.testing.assert_close(gradient, torch.zeros_like(original))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_balanced_reduction(dtype):
    small = torch.finfo(dtype).eps / 2
    values = torch.tensor([1.0, 0.0] + [small] * 6, dtype=dtype, requires_grad=True)
    expected = 1.0 + 6 * small
    terms = values.unbind()
    partials = []
    for value in terms:
        add_pairwise(partials, value)
    for result in (
        sum_pairwise(terms),
        sum_pairwise(value for value in partials if value is not None),
    ):
        torch.testing.assert_close(
            result, torch.as_tensor(expected, dtype=dtype), atol=0, rtol=0
        )
        torch.testing.assert_close(
            torch.autograd.grad(result, values)[0], torch.ones_like(values)
        )
