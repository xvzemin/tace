"""Aligned frame products through a complete force-loss parameter gradient."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import o2
from equivariantx.ops import config
from equivariantx.ops.torch.o2.o3_tensor_product import handwritten, reference


def training_problem(mode, backend, device, *, optimized=False):
    layer = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, mode, True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(device=device, dtype=torch.float64)
    frame = o2.WignerD(2, 2, method="recursive").to(device=device, dtype=torch.float64)
    values = (
        torch.randn(2, 6, device=device, dtype=torch.float64) * 0.1,
        torch.randn(2, 3, device=device, dtype=torch.float64) + 0.5,
        torch.randn(2, layer.weight_numel, device=device, dtype=torch.float64) * 0.1,
    )
    values = tuple(v.requires_grad_() for v in values)

    def run(x, r, w, ordinary=False):
        rotation, inverse = frame(r)
        if ordinary:
            y = reference(layer, layer.local_frame_in.to_local(x, rotation), w)
            y = layer.local_frame_out.to_global(y, inverse)
        elif optimized:
            y = handwritten(layer, layer.local_frame_in.to_local(x, rotation), w)
            y = layer.local_frame_out.to_global(y, inverse)
        else:
            y = layer(x, rotation, inverse, w)
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, r, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, torch.autograd.grad(loss, w)[0]

    return run, values


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
def test_joint_inductor(mode, monkeypatch, device):
    monkeypatch.setattr(config, "TORCH_ALIGNED_PRODUCT", True)
    run, values = training_problem(mode, "torch", device, optimized=True)
    graph = make_fx(lambda *args: run(*args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    x, r, w = values
    for _ in range(3):
        expected = run(x, r, w, True)
        torch.testing.assert_close(
            compiled(x.detach(), r.detach(), w.detach()), expected, atol=3e-9, rtol=3e-9
        )
        w = (w - 0.001 * expected[-1]).detach().requires_grad_()
