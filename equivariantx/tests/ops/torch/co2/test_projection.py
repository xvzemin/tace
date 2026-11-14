"""Compiler-visible planar polynomials and complete force-loss gradients."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config


@pytest.mark.parametrize("rank", [2, 3])
def test_joint_inductor(rank, monkeypatch):
    monkeypatch.setattr(config, "TORCH_TENSOR_AXIS", True)
    monkeypatch.setattr(config, "TORCH_PLANAR_DETRACING", True)
    monkeypatch.setattr(config, "TORCH_PLANAR_CUBIC", True)
    project = co2.PlaneProjector(rank)
    detrace = co2.PlanarDetracer(rank, dtype=torch.float64)
    x = (torch.randn(3, 2, 3**rank, dtype=torch.float64) * 0.2).requires_grad_()
    v = (torch.randn(3, 1, 3, dtype=torch.float64) + 0.5).requires_grad_()
    w = (torch.randn(2, 1, dtype=torch.float64) * 0.2).requires_grad_()

    def operation(x, v, w):
        p = co2.plane_projector(torch.nn.functional.normalize(v, dim=-1))
        y = detrace(project(x * w, p), p)
        e = y.sin().sum()
        force = -torch.autograd.grad(e, v, create_graph=True)[0]
        loss = force.square().sum() + e.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(operation)(x, v, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        expected = operation(x, v, w)
        actual = compiled(x.detach(), v.detach(), w.detach())
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
