"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import co2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_time_reversal_operators(backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    irreps = co2.Irreps("2x0ee+2x0eo+2x0oe+2x0oo+2x1me+2x1mo+2x2me+2x2mo")
    instructions = [
        (i, j, k, "uuu", True)
        for i, (_, a) in enumerate(irreps)
        for j, (_, b) in enumerate(irreps)
        for k, (_, c) in enumerate(irreps)
        if c in a * b
    ]
    tp = co2.TensorProduct(
        irreps, irreps, irreps, instructions, backend=backend, internal_weights=False
    ).to(DEVICE)
    x, y = [irreps.randn(5, -1, device=DEVICE).requires_grad_() for _ in range(2)]
    w = torch.randn(tp.weight_numel, device=DEVICE, requires_grad=True)
    time = irreps.D_from_angle(x.new_tensor(0), time_reversal=True)
    actual, expected = tp(x @ time.T, y @ time.T, w), tp(x, y, w) @ time.T
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        probe = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((z.sin() * probe).sum(), (x, y, w), create_graph=True)
            for z in (actual, expected)
        ]
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]
    linear = co2.Linear(irreps, irreps, biases=True).to(DEVICE)
    torch.testing.assert_close(linear(x @ time.T), linear(x) @ time.T)
    gate = co2.Gate(
        "2x0eo", [torch.tanh], "2x0oo", [torch.tanh], "2x1mo", backend=backend
    ).to(DEVICE)
    h = gate.irreps_in.randn(5, -1, device=DEVICE)
    din = gate.irreps_in.D_from_angle(h.new_tensor(0), time_reversal=True)
    dout = gate.irreps_out.D_from_angle(h.new_tensor(0), time_reversal=True)
    torch.testing.assert_close(gate(h @ din.T), gate(h) @ dout.T)
    with pytest.raises(ValueError):
        co2.TensorProduct(
            "1mo", "1me", "0ee", [(0, 0, 0, "uuu", True)], backend=backend
        )
