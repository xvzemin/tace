"""Joint force-loss parameter updates through the module's Torch implementation."""

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3
from equivariantx.ops import config


def training_problem(device="cpu", backend="torch"):
    torch.manual_seed(97)
    layer = (
        co3.ictp.TensorProduct(
            "2x2e",
            "2x2e",
            "2x2e",
            [(0, 0, 0, "uuu", True)],
            backend=backend,
            internal_weights=False,
        )
        .double()
        .to(device)
    )
    x = (torch.randn(3, 18, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    y = (torch.randn(3, 18, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    w = (torch.randn(2, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    def operation(x, y, w):
        z = layer(x, y, w)
        energy = 1 + z.sin().sum()
        force = -torch.autograd.grad(energy, y, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, torch.autograd.grad(loss, w)[0]

    return operation, (x, y, w)


def test_joint_inductor(monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    operation, values = training_problem()
    graph = make_fx(operation)(*values)
    run = torch.compile(graph, fullgraph=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    x, y, w = values
    for _ in range(3):
        expected = operation(x, y, w)
        actual = run(x.detach(), y.detach(), w.detach())
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()


def test_literal_angular_joint(monkeypatch):
    from equivariantx.co3.ictp.tensor_product import CartesianProduct
    from equivariantx.ops.torch.co3.ictp.tensor_product import reference

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    for degrees in ((1, 1, 2), (2, 1, 2), (3, 1, 3)):
        layer = CartesianProduct(*degrees).double()
        assert layer.literal_entries
        x = (
            torch.randn(2, 2, 3 ** degrees[0], dtype=torch.float64) * 0.1
        ).requires_grad_()
        y = (
            torch.randn(2, 2, 3 ** degrees[1], dtype=torch.float64) * 0.1
        ).requires_grad_()
        w = (torch.randn(2, 1, dtype=torch.float64) * 0.1).requires_grad_()

        def operation(x, y, w, ordinary=False):
            z = reference(layer, x * w, y) if ordinary else layer(x * w, y)
            energy = 1 + z.sin().sum()
            force = -torch.autograd.grad(energy, y, create_graph=True)[0]
            loss = energy.square() + force.square().sum()
            return loss, force, torch.autograd.grad(loss, w)[0]

        graph = make_fx(lambda x, y, w: operation(x, y, w))(x, y, w)
        run = torch.compile(graph, fullgraph=True)
        for _ in range(3):
            expected = operation(x, y, w, True)
            torch.testing.assert_close(
                run(x.detach(), y.detach(), w.detach()),
                expected,
                atol=3e-10,
                rtol=3e-10,
            )
            w = (w - 0.01 * expected[-1]).detach().requires_grad_()
