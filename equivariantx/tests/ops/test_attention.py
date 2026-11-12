"""Native segmented normalization with arbitrary functional derivatives."""

import pytest
import torch

from equivariantx.conv.attention import graph_softmax
from equivariantx.ops import get_kernel
from equivariantx.ops.attention import evaluate_adjoint
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph attention")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
@pytest.mark.parametrize("edges,weight_heads", [(0, 1), (7, 1), (7, 3)])
def test_functional_transforms(compiler, edges, weight_heads, double_precision):
    values = (
        torch.randn(edges, 3, device="cuda"),
        torch.rand(edges, weight_heads, device="cuda") + 0.5,
        torch.arange(edges, device="cuda") % 3,
    )

    def function(scores, weight, target):
        return graph_softmax(scores, target, 5, weight, 1e-5, backend="cuda")

    def reference(scores, weight, target):
        return graph_softmax(scores, target, 5, weight, 1e-5, backend="torch")

    assert_functional_derivatives(function, reference, values, compiler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph attention")
@pytest.mark.parametrize("weight_heads", [1, 3])
def test_adjoint_and_four_derivatives(weight_heads, double_precision):
    values = tuple(
        torch.randn(7, width, device="cuda", requires_grad=True) * 0.2
        for width in (3, 3, weight_heads)
    )
    target = torch.arange(7, device="cuda") % 3
    actual = evaluate_adjoint(*values, target, 5)
    expected = get_kernel("graph_normalized_exp_adjoint", "torch")(*values, target, 5)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    actual, expected = [
        sum(value.sin().sum() for value in output) for output in (actual, expected)
    ]
    for _ in range(4):
        gradients = [
            torch.autograd.grad(value, values, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [sum(value.sum() for value in grads) for grads in gradients]
