"""Exact sparse bilinear maps and arbitrary mixed derivatives."""

import pytest
import torch

from equivariantx.ops.common.sparse import bilinear_rows
from equivariantx.ops.common.torch.sparse_bilinear import evaluate, primal, reference


def coefficient_tables(device="cpu", dtype=torch.float64, dense=False):
    generator = torch.Generator().manual_seed(17)
    tensor = torch.randn(3, 4, 5, generator=generator, dtype=torch.float64)
    if not dense:
        tensor = tensor * (tensor.abs() > 1.1)
    tables = tuple(
        value.to(
            device=device, dtype=dtype if value.is_floating_point() else torch.long
        )
        for role in range(3)
        for value in bilinear_rows(
            tensor.permute(*(i for i in range(3) if i != role), role)
        )
    )
    return tensor.to(device=device, dtype=dtype), tables


def check_derivatives(operation, device, role, dense=False):
    torch.manual_seed(19)
    tensor, tables = coefficient_tables(device, dense=dense)
    axes = [i for i in range(3) if i != role]
    x = (
        torch.randn(2, 1, 2 * tensor.shape[axes[0]], dtype=torch.float64, device=device)
        * 0.1
    )[..., ::2].requires_grad_()
    y = (
        torch.randn(1, 3, tensor.shape[axes[1]], dtype=torch.float64, device=device)
        * 0.1
    ).requires_grad_()

    def oracle(x, y):
        return torch.einsum("...i,...j,ijk->...k", x, y, tensor.permute(*axes, role))

    actual, expected = operation(x, y, tables, role), oracle(x, y)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    for _ in range(5):
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        losses = [(value.sin() * probe).sum() for value in (actual, expected)]
        actual, expected = [
            torch.cat(
                [
                    v.flatten()
                    for v in torch.autograd.grad(value, (x, y), create_graph=True)
                ]
            )
            for value in losses
        ]
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)

    def function(a, b):
        return operation(a, b, tables, role)

    tangents = tuple(torch.randn_like(value) for value in (x, y))
    torch.testing.assert_close(
        torch.func.jvp(function, (x, y), tangents),
        torch.func.jvp(oracle, (x, y), tangents),
    )
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(value, tangent)
            for value, tangent in zip((x, y), tangents)
        )
        result = torch.autograd.forward_ad.unpack_dual(function(*duals))
    torch.testing.assert_close(
        (result.primal, result.tangent), torch.func.jvp(oracle, (x, y), tangents)
    )
    torch.testing.assert_close(
        torch.vmap(function)(x[:, 0], y.expand(2, -1, -1)),
        torch.vmap(oracle)(x[:, 0], y.expand(2, -1, -1)),
    )
    empty = operation(x[:0], y, tables, role)
    assert empty.shape == (0, 3, tensor.shape[role])


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("operation", [evaluate, primal])
@pytest.mark.parametrize("dense", [False, True])
def test_sparse_bilinear(role, operation, dense):
    check_derivatives(operation, "cpu", role, dense)


def test_gradcheck():
    _, tables = coefficient_tables()
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)

    def function(x, y):
        return evaluate(x, y, tables)

    assert torch.autograd.gradcheck(function, (x, y))
    assert torch.autograd.gradgradcheck(function, (x, y))


def test_trainable_coefficients():
    _, tables = coefficient_tables()
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    coefficient = tables[-1].clone().requires_grad_()

    def function(x, y, c):
        return evaluate(x, y, (*tables[:-1], c))

    assert torch.autograd.gradcheck(function, (x, y, coefficient))
    assert torch.autograd.gradgradcheck(function, (x, y, coefficient))
    expected = reference(x, y, (*tables[:-1], coefficient))
    torch.testing.assert_close(function(x, y, coefficient), expected)


def check_coefficient_forward_ad(operation, device, role, trainable):
    from equivariantx.tests.utils.testing import check_forward_ad

    tensor, tables = coefficient_tables(device)
    axes = [i for i in range(3) if i != role]
    x = torch.randn(
        2, tensor.shape[axes[0]], device=device, dtype=torch.float64
    ).requires_grad_(trainable)
    y = torch.randn(
        1, tensor.shape[axes[1]], device=device, dtype=torch.float64
    ).requires_grad_(trainable)
    coefficient = tables[3 * role + 2].clone().requires_grad_(trainable)

    def replaced(c):
        return (*tables[: 3 * role + 2], c, *tables[3 * role + 3 :])

    check_forward_ad(
        lambda x, y, c: operation(x, y, replaced(c), role),
        lambda x, y, c: reference(x, y, replaced(c), role),
        (x, y, coefficient),
    )


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("trainable", [False, True])
def test_coefficient_forward_ad(role, trainable):
    check_coefficient_forward_ad(evaluate, "cpu", role, trainable)


@pytest.mark.parametrize("dimensions", [(0, 3, 2), (2, 0, 3), (2, 3, 0)])
def test_empty_components(dimensions):
    from equivariantx.ops.common.sparse_coefficients import bilinear_tables

    tables = bilinear_tables(dimensions, ())
    x = torch.empty(2, dimensions[0], dtype=torch.float64, requires_grad=True)
    y = torch.empty(1, dimensions[1], dtype=torch.float64, requires_grad=True)
    z = evaluate(x, y, tables)
    torch.testing.assert_close(z, torch.zeros(2, dimensions[2], dtype=torch.float64))
    dx, dy = torch.autograd.grad(z.sum(), (x, y))
    torch.testing.assert_close(dx, torch.zeros_like(x))
    torch.testing.assert_close(dy, torch.zeros_like(y))
