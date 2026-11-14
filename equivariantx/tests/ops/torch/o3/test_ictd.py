"""Complete Cartesian gradients through the joint Torch compiler."""

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import config
from equivariantx.tests.utils.testing import ictd_training_problem


def test_joint_inductor(monkeypatch, device):
    monkeypatch.setattr(config, "TORCH_ICTD", True)
    monkeypatch.setattr(config, "TORCH_ICTD_ADJOINT", True)
    monkeypatch.setattr(config, "ICTD_DENSE_MAX_WIDTH", 0)
    module, (x, w), run = ictd_training_problem(3, 3, "torch", device)
    graph = make_fx(run)(x, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        actual = compiled(x.detach(), w.detach())
        expected = run(x, w)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
