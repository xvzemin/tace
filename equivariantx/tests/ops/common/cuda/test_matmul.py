"""Native matrix products, functional AD and capture."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.cuda.matmul import matmul
from equivariantx.ops.common.torch.matmul import matmul as ordinary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows,inner", [(0, 7), (3, 0), (3, 7), (5, 33)])
def test_derivatives(dtype, rows, inner):
    a = (
        torch.randn(2, inner, rows, device="cuda", dtype=dtype).transpose(-1, -2) * 0.1
    ).requires_grad_()
    b = (
        torch.randn(1, 5, inner, device="cuda", dtype=dtype).transpose(-1, -2) * 0.1
    ).requires_grad_()
    actual, expected = matmul(a, b), ordinary(a, b)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        gradients = [
            torch.autograd.grad(value.sin().sum(), (a, b), create_graph=True)
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([part.flatten() for part in result]) * 0.1 for result in gradients
        ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    graph = make_fx(matmul)(a, b)
    torch.testing.assert_close(
        graph(a, b), ordinary(a, b), atol=tolerance, rtol=tolerance
    )


def test_functional_transforms():
    a = torch.randn(2, 3, 7, device="cuda") * 0.1
    b = torch.randn(1, 7, 5, device="cuda") * 0.1
    da, db = torch.randn_like(a) * 0.1, torch.randn_like(b) * 0.1
    results = []
    for function in (matmul, ordinary):
        gradient = torch.func.grad(
            lambda a, b: function(a, b).sin().sum(), argnums=(0, 1)
        )
        results.append(
            torch.vmap(lambda a: torch.func.jvp(gradient, (a, b), (da, db)))(
                torch.stack((a, a * 0.8))
            )
        )
    torch.testing.assert_close(*results, atol=3e-6, rtol=3e-5)
    nested = []
    for function in (matmul, ordinary):

        def tangent(a, b):
            return torch.func.jvp(function, (a, b), (da, db))[1]

        nested.append(torch.func.jvp(tangent, (a, b), (da, db)))
    torch.testing.assert_close(*nested, atol=3e-6, rtol=3e-5)
    for function in (matmul, ordinary):
        with torch.autograd.forward_ad.dual_level():
            value = function(torch.autograd.forward_ad.make_dual(a, da), b)
            actual = torch.autograd.forward_ad.unpack_dual(value)
        expected = torch.func.jvp(ordinary, (a, b), (da, torch.zeros_like(b)))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
