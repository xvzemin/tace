"""Sparse linear chains preserve arbitrary reverse and forward derivatives."""

import pytest
import torch

from equivariantx.ops.common.sparse import matrix_rows
from equivariantx.ops.common.torch.sparse_chain import evaluate


@pytest.mark.parametrize("reverse", [False, True])
def test_derivatives(reverse, double_precision):
    a = torch.randn(3, 5) * 0.2
    b = torch.randn(4, 6) * 0.2
    tables = (*matrix_rows(a), *matrix_rows(a.T), *matrix_rows(b), *matrix_rows(b.T))
    x = (torch.randn(2, 3, 8 if reverse else 20) * 0.1)[..., ::2].requires_grad_()

    def operation(value):
        return evaluate(value, (2, 1), reverse, *tables)

    def reference(value):
        if reverse:
            return ((value @ b).unflatten(-1, (2, 3)) @ a).flatten(-2)
        return ((value.unflatten(-1, (2, 5)) @ a.T).flatten(-2)) @ b.T

    actual, expected = operation(x), reference(x)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        actual, expected = [
            torch.autograd.grad(y.sin().sum(), x, create_graph=True)[0] * 0.1
            for y in (actual, expected)
        ]
    tangent = torch.randn_like(x)
    torch.testing.assert_close(
        torch.func.jvp(operation, (x,), (tangent,)),
        torch.func.jvp(reference, (x,), (tangent,)),
        atol=3e-11,
        rtol=3e-11,
    )
    torch.testing.assert_close(torch.vmap(operation)(x), operation(x))
    assert torch.autograd.gradcheck(operation, (x,))
    assert torch.autograd.gradgradcheck(lambda value: operation(value).sin(), (x,))
    assert operation(x[:0]).shape[0] == 0


def test_coefficients_and_tape(double_precision):
    a = torch.randn(3, 5) * 0.2
    tables = (*matrix_rows(a), *matrix_rows(a.T))
    x = torch.randn(2, 5, requires_grad=True) * 0.1
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        evaluate(x, (1,), False, *tables)
    assert all(value.data_ptr() != x.data_ptr() for value in saved)
    w = tables[1].clone().requires_grad_()

    def run(x, w):
        return evaluate(x, (1,), False, tables[0], w, *tables[2:])

    assert torch.autograd.gradcheck(run, (x, w))
    assert torch.autograd.gradgradcheck(lambda x, w: run(x, w).sin(), (x, w))
    directions = (torch.randn_like(x), torch.randn_like(w))
    from equivariantx.ops.common.torch.sparse_axis import reference

    torch.testing.assert_close(
        torch.func.jvp(run, (x, w), directions),
        torch.func.jvp(lambda x, w: reference(x, tables[0], w), (x, w), directions),
        atol=3e-11,
        rtol=3e-11,
    )
