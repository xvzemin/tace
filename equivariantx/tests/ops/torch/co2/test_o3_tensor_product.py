"""Joint force-loss parameter updates through the module's Torch implementation."""

import torch

from equivariantx import co2, co3
from equivariantx.tests.ops.testing import assert_joint_training


def training_problem(device="cpu", backend="torch"):
    torch.manual_seed(97)
    layer = (
        co2.O3TensorProduct(
            "2x2e",
            "3o",
            "2x2o",
            [(0, 0, 0, "uvu", True)],
            input_basis="cartesian",
            output_basis="cartesian",
            backend=backend,
            internal_weights=False,
        )
        .double()
        .to(device)
    )
    embedding = co3.ChangeOfBasis("2x2e").double().to(device)
    x = (torch.randn(3, 10, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    y = (torch.randn(3, 3, device=device, dtype=torch.float64) + 0.5).requires_grad_()
    w = (torch.randn(2, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    def operation(x, y, w):
        z = layer(embedding(x), y, w)
        energy = 1 + z.sin().sum()
        force = -torch.autograd.grad(energy, y, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, torch.autograd.grad(loss, w)[0]

    return operation, (x, y, w)


def test_joint_inductor(monkeypatch):
    assert_joint_training(training_problem, monkeypatch)
