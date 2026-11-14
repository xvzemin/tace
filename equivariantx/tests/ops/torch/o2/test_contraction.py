"""Joint force-loss compilation for the corresponding O(2) operator."""

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import config
from equivariantx.tests.o2.testing import operator_training_problem


def test_joint_training(monkeypatch):
    monkeypatch.setattr(config, "TORCH_O2_CONTRACTION", True)
    operation, values = operator_training_problem("contraction", "torch", "cpu")
    graph = make_fx(lambda *args: operation(*args))(*values)
    compiled = torch.compile(graph, fullgraph=True)
    x, w = values
    for _ in range(3):
        expected = operation(x, w, True)
        with torch.no_grad():
            actual = compiled(x.detach(), w.detach())
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
