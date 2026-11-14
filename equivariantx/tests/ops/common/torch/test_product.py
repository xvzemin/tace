"""Sparse product constants across independent functional transforms."""

import pytest
import torch

from equivariantx.ops.common.torch.product import (
    product,
    product_indices,
    unrolled_product,
)


@pytest.mark.parametrize("role", range(3))
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("required", [0, 7])
def test_bilinear_transposes_and_five_orders(role, shared, required, monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    dimensions, width = (3, 2, 4), 3
    metadata = (
        width,
        dimensions,
        (((0, 0, 0), 0.3), ((1, 0, 2), -0.7), ((2, 1, 1), 0.2), ((1, 1, 2), 0.6)),
    )
    values = tuple(
        (
            torch.randn(1 if shared else 3, d * width, dtype=torch.float64) * 0.1
        ).requires_grad_()
        for d in dimensions
    )
    active = tuple(v for i, v in enumerate(values) if i != role)

    def function(*values):
        return product(metadata, role, required, values, 3)

    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = function(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = make_fx(function)(*values)(*values)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(5):
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        gradients = [
            torch.autograd.grad((v.sin() * probe).sum(), active, create_graph=True)
            for v in (actual, expected)
        ]
        actual, expected = (torch.cat([v.flatten() for v in gs]) for gs in gradients)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("role", range(3))
@pytest.mark.parametrize("required", [0, 1, 3, 7])
def test_bilinear_missing_roles_preserve_mask(role, required):
    metadata = (
        2,
        (2, 2, 2),
        (((0, 0, 0), 0.5), ((1, -1, 1), -0.3), ((-1, -1, 0), 0.2)),
    )
    values = tuple(torch.randn(3, 4, dtype=torch.float64) for _ in range(3))
    torch.testing.assert_close(
        product(metadata, role, required, values, 3),
        unrolled_product(metadata, role, required, values, 3),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("role", range(3))
def test_bilinear_wide_gpu_transform_constants(role, monkeypatch):
    from equivariantx.ops import config

    entries = tuple(
        ((a, b, c), (1 + a - b + c) / 100)
        for a in range(7)
        for b in range(7)
        for c in range(7)
    )
    metadata = (2, (7, 7, 7), entries)
    values = tuple(
        (torch.randn(3, 14, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
        for _ in range(3)
    )
    directions = tuple(torch.randn_like(v) * 0.1 for v in values)

    def function(*values):
        return product(metadata, role, 0, values, values[0].shape[0])

    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = torch.func.jvp(function, values, directions)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    for _ in range(2):
        torch.testing.assert_close(
            torch.func.jvp(function, values, directions),
            expected,
            atol=2e-12,
            rtol=2e-12,
        )
        torch.testing.assert_close(
            function(*values), expected[0], atol=2e-12, rtol=2e-12
        )
    torch.testing.assert_close(function(*(v[:0] for v in values)), expected[0][:0])


def test_functional_constants_do_not_escape(double_precision):
    product_indices.cache_clear()
    metadata = (2, (2, 2, 2), (((0, 0, 0), 0.3), ((1, 1, 1), -0.2)))
    values = tuple(torch.randn(3, 4) for _ in range(2))
    tangents = tuple(torch.randn_like(value) for value in values)

    def function(x, y):
        return product(metadata, 2, 0, (x, y, x), x.shape[0])

    def reference(x, y):
        return x * y * x.new_tensor((0.3, 0.3, -0.2, -0.2))

    for scale in (1.0, 0.7, 1.0):
        inputs = tuple(value * scale for value in values)
        torch.testing.assert_close(
            torch.func.jvp(function, inputs, tangents),
            torch.func.jvp(reference, inputs, tangents),
        )
        torch.testing.assert_close(function(*inputs), reference(*inputs))
    product_indices.cache_clear()


def product_case(role, rows=3, shared=False, required=0, device="cpu"):
    width, dims = 2, (2, 2, 1, 3)
    metadata = (
        width,
        dims,
        (
            ((0, 0, 0, 0), 0.3),
            ((1, 0, 0, 1), -0.2),
            ((0, 1, 0, 1), 0.7),
            ((1, 1, 0, 1), 0.4),
            ((-1, -1, -1, 2), 0.5),
        ),
    )
    values = tuple(
        (
            torch.randn(1 if shared else rows, dim * width, device=device) * 0.1
        ).requires_grad_()
        for dim in dims
    )

    def reference(*values):
        expanded = tuple(x.expand(rows, -1) for x in values)

        def primal(x, y, gate):
            x0, x1 = x.split(width, -1)
            y0, y1 = y.split(width, -1)
            zero = x[:, :width] * 0
            return torch.cat(
                (
                    0.3 * x0 * y0 * gate,
                    (-0.2 * x1 * y0 + 0.7 * x0 * y1 + 0.4 * x1 * y1) * gate,
                    zero if required else zero + 0.5,
                ),
                dim=-1,
            )

        if role == 3:
            return primal(*expanded[:3])
        return torch.func.grad(
            lambda *args: (primal(*args[:3]) * args[3]).sum(), argnums=role
        )(*expanded)

    return metadata, values, reference


@pytest.mark.parametrize("role", range(4))
@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("required", [0, 7])
def test_unrolled_transposes(role, rows, shared, required, double_precision):
    metadata, values, reference = product_case(role, rows, shared, required)
    actual = unrolled_product(metadata, role, required, values, rows)
    expected = reference(*values)
    active = tuple(value for i, value in enumerate(values) if i != role)
    directions = tuple(torch.randn_like(value) * 0.1 for value in active)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual, expected = (value.sin().sum() for value in (actual, expected))
    for _ in range(4):
        gradients = [
            torch.autograd.grad(value, active, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-11, rtol=1e-11)
        actual, expected = [
            sum((g * d).sum() for g, d in zip(gs, directions)) for gs in gradients
        ]


@pytest.mark.parametrize("role", [0, 3])
def test_unrolled_functional_compilation(role, double_precision):
    from equivariantx.tests.ops.testing import assert_functional_derivatives

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata, values, reference = product_case(role, device=device)

    def function(*args):
        return product(metadata, role, 0, args[:-1], args[-1].shape[0])

    assert_functional_derivatives(
        function,
        lambda *args: reference(*args[:-1]),
        (*values, torch.arange(3, device=device)),
        compiler="inductor",
    )
