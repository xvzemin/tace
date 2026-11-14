"""Fixed spherical transforms and their complete joint gradients."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.o3.s2grid import S2Grid
from equivariantx.ops import config
from equivariantx.ops.torch.o3 import s2grid as kernels


@pytest.mark.parametrize("quadrature", ["lebedev", "gauss_legendre", "equiangular"])
def test_joint_inductor(quadrature, monkeypatch, device):
    monkeypatch.setattr(config, "TORCH_S2_GRID", True)
    layer = S2Grid(2, quadrature=quadrature, dtype=torch.float64, device=device)
    x = (
        torch.randn(2, layer.dim, dtype=torch.float64, device=device) * 0.1
    ).requires_grad_()
    w = (
        torch.randn(layer.dim, dtype=torch.float64, device=device) * 0.1
    ).requires_grad_()

    def run(x, w, ordinary=False):
        to_grid = kernels.to_grid_reference if ordinary else kernels.to_grid
        from_grid = kernels.from_grid_reference if ordinary else kernels.from_grid
        y = from_grid(layer, to_grid(layer, x * w).sin())
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(lambda *args: run(*args))(x, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    for _ in range(3):
        expected = run(x, w, True)
        actual = compiled(x.detach(), w.detach())
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        w = (w - 0.001 * expected[-1]).detach().requires_grad_()
