"""Broadcast tensor-axis values, exact derivatives and functional transforms."""

import pytest
import torch

from equivariantx.ops.common.torch.tensor_axis import evaluate, reference


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("matrix_shape", [(2, 1, 3, 3), (3, 3, 3), (3, 3)])
def test_tensor_axes(backend, matrix_shape, device):
    function = evaluate
    if backend == "cuda" and device == "cuda":
        from equivariantx.ops.common.cuda.tensor_axis import evaluate as function
    x = (
        torch.randn(2, 3, 27, device=device, dtype=torch.float64) * 0.2
    ).requires_grad_()
    matrix = (
        torch.randn(*matrix_shape, device=device, dtype=torch.float64) * 0.2
    ).requires_grad_()
    values = x, matrix
    actual, expected = function(x, matrix, 3, 3), reference(x, matrix, 3, 3)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    losses = [v.sin().sum() for v in (actual, expected)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
        tangents = tuple(torch.randn_like(v) * 0.1 for v in values)
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
    assert torch.autograd.gradcheck(
        lambda a, b: function(a, b, 3, 3), values, fast_mode=True
    )
    assert torch.autograd.gradgradcheck(
        lambda a, b: function(a, b, 3, 3), values, fast_mode=True
    )


def test_torch_functional_axis(device):
    x = torch.randn(2, 3, 27, device=device, dtype=torch.float64)
    matrix = torch.randn(2, 1, 3, 3, device=device, dtype=torch.float64)

    def a(x, m):
        return evaluate(x, m, 3, 3)

    def b(x, m):
        return reference(x, m, 3, 3)

    tangent = torch.randn_like(x), torch.randn_like(matrix)
    torch.testing.assert_close(
        torch.func.jvp(a, (x, matrix), tangent),
        torch.func.jvp(b, (x, matrix), tangent),
        atol=3e-11,
        rtol=3e-11,
    )

    def grad(f):
        return torch.func.grad(lambda x, m: f(x, m).sin().sum(), argnums=(0, 1))

    torch.testing.assert_close(
        torch.func.jvp(grad(a), (x, matrix), tangent),
        torch.func.jvp(grad(b), (x, matrix), tangent),
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(
        torch.vmap(a)(x, matrix), a(x, matrix), atol=3e-11, rtol=3e-11
    )
