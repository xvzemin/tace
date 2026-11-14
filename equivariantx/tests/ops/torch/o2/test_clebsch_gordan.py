"""Joint force-loss compilation for the corresponding O(2) operator."""

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import config
from equivariantx.tests.o2.testing import operator_training_problem


def test_joint_training(monkeypatch):
    monkeypatch.setattr(config, "TORCH_O2_COUPLING", True)
    operation, values = operator_training_problem("clebsch_gordan", "torch", "cpu")
    graph = make_fx(lambda *args: operation(*args))(*values)
    compiled = torch.compile(graph, fullgraph=True)
    x, w = values
    for _ in range(3):
        expected = operation(x, w, True)
        with torch.no_grad():
            actual = compiled(x.detach(), w.detach())
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()


def test_dynamic_short_coupling(monkeypatch):
    from equivariantx import o2
    from equivariantx.o2.clebsch_gordan import clebsch_gordan_product
    from equivariantx.ops.torch.o2.clebsch_gordan import reference

    monkeypatch.setattr(config, "TORCH_O2_COUPLING", True)
    a, b, out = o2.Irrep("1m"), o2.Irrep("2m"), o2.Irrep("1m")
    weight = torch.randn(1, 2, 4, dtype=torch.float64)

    def call(x):
        return clebsch_gordan_product(x, a, weight, b, out, elementwise=True)

    compiled = torch.compile(call, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, 2, 4, dtype=torch.float64)
        torch.testing.assert_close(
            compiled(x), reference(x, a, weight, b, out, elementwise=True)
        )
