"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx import conv as eqx_conv


@pytest.mark.parametrize("edges", [0, 7])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("shared", [False, True])
def test_wigner6j_convolution(edges, backend, shared, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = eqx_conv.Wigner6jTensorProductConv(
        "2x1o",
        "1o",
        "1e",
        "2x1e",
        "2x0e+2x1e+2x2e",
        [(0, 0, 0, "uvu", True)],
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=shared,
        backend=backend,
    ).to(device)
    inputs = [
        torch.randn(*shape, device=device, requires_grad=True)
        for shape in (
            (4, 6),
            (edges, 3),
            (4, 3),
            (2,) if shared else (edges, 2),
            (6,) if shared else (edges, 6),
        )
    ]
    x1, x2, x3, w1, w2 = inputs
    index = torch.randint(3, (2, edges), device=device)
    expected = module.forward_reference(x1, x2, x3, index, w1, w2)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for evaluate in (module, compiled):
        actual = evaluate(x1, x2, x3, index, w1, w2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    exported = torch.export.export(module, (x1, x2, x3, index, w1, w2))
    torch.testing.assert_close(
        exported.module()(x1, x2, x3, index, w1, w2),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
