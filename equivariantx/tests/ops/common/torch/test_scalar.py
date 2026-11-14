"""Scalar geometry saves operands while retaining arbitrary Torch derivatives."""

import torch

from equivariantx.ops.common.torch.scalar import evaluate, primal


def test_scalar_adjoint_and_saved_tape():
    x = (torch.randn(3, 3, dtype=torch.float64) * 0.1 + 0.7).requires_grad_()

    def function(v):
        return evaluate(1, 1e-7, False, (), v)[0]

    assert torch.autograd.gradcheck(function, (x,))
    assert torch.autograd.gradgradcheck(function, (x,))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda v: (saved.append(v), v)[1], lambda v: v
    ):
        function(x)
    assert len(saved) == 1 and saved[0] is x
    a, b = function(x), primal(1, 1e-7, False, (), x)[0]
    for _ in range(5):
        probe = torch.randn_like(a) / a.numel() ** 0.5
        a, b = [
            torch.autograd.grad((v.sin() * probe).sum(), x, create_graph=True)[0]
            for v in (a, b)
        ]
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
