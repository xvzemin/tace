"""Restrictions and lifts in complete joint force-loss parameter gradients."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config


@pytest.mark.parametrize(
    "rank,device",
    [(2, "cpu"), (3, "cpu"), (4, "cpu"), (6, "cpu"), (2, "cuda"), (4, "cuda")],
)
def test_joint_inductor(rank, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_RESTRICTION", True)
    layer = co2.Restriction(rank).to(device=device, dtype=torch.float64)
    projector = co2.TransverseProjector(rank).to(device=device, dtype=torch.float64)
    x = (
        torch.randn(2, 2, 3**rank, device=device, dtype=torch.float64) * 0.1
    ).requires_grad_()
    v = (
        torch.randn(2, 1, 3, device=device, dtype=torch.float64) + 0.5
    ).requires_grad_()
    w = (torch.randn(2, 1, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    def run(x, v, w):
        n = torch.nn.functional.normalize(v, dim=-1)
        parts = tuple(p.sin() for p in layer(x * w, n))
        y = layer.inverse(parts, n) + projector(x * w, n)
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(run)(x, v, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        actual = compiled(x.detach(), v.detach(), w.detach())
        expected = run(x, v, w)
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()


@pytest.mark.parametrize("rank", [3, 6])
def test_cpu_dynamic(rank, monkeypatch):
    layer = co2.Restriction(rank).double()
    projector = co2.TransverseProjector(rank).double()

    def operation(x, direction):
        n = torch.nn.functional.normalize(direction, dim=-1)
        return *layer(x, n), projector(x, n)

    run = torch.compile(operation, backend="aot_eager", fullgraph=True, dynamic=True)
    for rows in (2, 5, 0):
        x = torch.randn(rows, 2, 3**rank, dtype=torch.float64) * 0.1
        direction = torch.randn(rows, 1, 3, dtype=torch.float64) + 0.5
        actual = run(x, direction)
        with monkeypatch.context() as context:
            context.setattr(config, "TORCH_REFERENCE", True)
            expected = operation(x, direction)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
