"""Sparse rows, broadcast adjoints and higher functional derivatives."""

import pytest
import torch

from equivariantx.ops.common.torch.sparse_axis import evaluate, reference


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("weight_shape", [(2, 1, 4, 3), (3, 4, 3), (4, 3)])
def test_sparse_axes(backend, weight_shape, device):
    run = evaluate
    if backend == "cuda" and device == "cuda":
        from equivariantx.ops.common.cuda.sparse_axis import evaluate as run
    indices = torch.tensor([[0, 1, 2], [2, 4, 4], [3, 0, 3], [1, 2, 4]], device=device)
    x = (torch.randn(2, 3, 60, device=device, dtype=torch.float64) * 0.1)[
        ..., ::2
    ].requires_grad_()
    w = (
        torch.randn(*weight_shape, device=device, dtype=torch.float64) * 0.1
    ).requires_grad_()
    actual, expected = run(x, indices, w, 2, 3), reference(x, indices, w, 2, 3)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    losses = [v.sin().sum() for v in (actual, expected)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, (x, w), create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
        tangents = torch.randn_like(x) * 0.1, torch.randn_like(w) * 0.1
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
    assert torch.autograd.gradcheck(
        lambda a, b: run(a, indices, b, 2, 3), (x, w), fast_mode=True
    )
    assert torch.autograd.gradgradcheck(
        lambda a, b: run(a, indices, b, 2, 3), (x, w), fast_mode=True
    )


def test_functional_axes(device):
    indices = torch.tensor([[0, 1, 2], [2, 4, 4]], device=device)
    x = torch.randn(2, 3, 30, device=device, dtype=torch.float64)
    w = torch.randn(2, 1, 2, 3, device=device, dtype=torch.float64)

    def actual(x, w):
        return evaluate(x, indices, w, 2, 3)

    def expected(x, w):
        return reference(x, indices, w, 2, 3)

    tangents = torch.randn_like(x), torch.randn_like(w)
    torch.testing.assert_close(
        torch.func.jvp(actual, (x, w), tangents),
        torch.func.jvp(expected, (x, w), tangents),
        atol=3e-11,
        rtol=3e-11,
    )

    def grad(f):
        return torch.func.grad(lambda x, w: f(x, w).sin().sum(), argnums=(0, 1))

    torch.testing.assert_close(
        torch.func.jvp(grad(actual), (x, w), tangents),
        torch.func.jvp(grad(expected), (x, w), tangents),
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(
        torch.vmap(actual)(x, w), actual(x, w), atol=3e-11, rtol=3e-11
    )
