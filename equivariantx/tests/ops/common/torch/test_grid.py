"""Shared grid adjoints retain constants and support nested differentiation."""

import pytest
import torch

from equivariantx.ops.common.torch.grid import evaluate, linear, primal


@pytest.mark.parametrize("sheets", [2, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_sheet_transpose(sheets, reverse):
    width, channels = 3, 2
    size = sheets * width * channels
    order = torch.randperm(size)
    inverse = order.argsort()
    sign = torch.randint(0, 2, (size,), dtype=torch.int8) * 2 - 1
    output_sign = sign[inverse]
    x = (torch.randn(2, size, dtype=torch.float64) * 0.1).requires_grad_()
    tables = order, inverse, sign, output_sign, sheets, width, reverse
    actual, expected = evaluate(x, *tables).sin(), primal(x, *tables).sin()
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        actual, expected = [
            torch.autograd.grad(v.sin().sum(), x, create_graph=True)[0]
            for v in (actual, expected)
        ]
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        evaluate(x, *tables)
    assert all(value is not x for value in saved)

    def operation(value):
        return evaluate(value, *tables).sin()

    tangent = torch.randn_like(x)
    torch.testing.assert_close(
        torch.func.jvp(operation, (x,), (tangent,)),
        torch.autograd.functional.jvp(operation, x, tangent),
    )
    torch.testing.assert_close(torch.vmap(operation)(x), operation(x))
    assert torch.autograd.gradcheck(operation, (x,))
    assert torch.autograd.gradgradcheck(operation, (x,))


def test_trainable_grid_matrix():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(linear, (x, matrix))
    assert torch.autograd.gradgradcheck(linear, (x, matrix))
    tangent = torch.randn_like(x), torch.randn_like(matrix)
    torch.testing.assert_close(
        torch.func.jvp(linear, (x, matrix), tangent),
        torch.func.jvp(lambda x, w: x @ w.T, (x, matrix), tangent),
    )


def test_dual_constant_matrix():
    x = torch.randn(2, 3, dtype=torch.float64)
    matrix = torch.randn(4, 3, dtype=torch.float64)
    dx, dw = torch.randn_like(x), torch.randn_like(matrix)
    with torch.autograd.forward_ad.dual_level():
        primal, tangent = torch.autograd.forward_ad.unpack_dual(
            linear(
                torch.autograd.forward_ad.make_dual(x, dx),
                torch.autograd.forward_ad.make_dual(matrix, dw),
            )
        )
    assert tangent is not None
    torch.testing.assert_close(primal, x @ matrix.T)
    torch.testing.assert_close(tangent, dx @ matrix.T + x @ dw.T)
