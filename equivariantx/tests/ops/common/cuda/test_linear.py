"""Compiler-visible indexed maps and force-training adjoints."""

import pytest
import torch
from e3nn import o3

from equivariantx.o3 import MoEElementLinear
from equivariantx.ops import config
from equivariantx.ops.common.linear import contract
from equivariantx.ops.common.torch.linear import contract_reference
from equivariantx.ops.torch.nn.element_linear import reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


def module(layout, wide=False):
    inputs, outputs = (
        ("65x0e+17x1o", "67x0e+19x1o") if wide else ("3x0e+2x0e+5x1o", "4x0e+7x1o+1x2e")
    )
    value = MoEElementLinear(
        o3.Linear(inputs, outputs, internal_weights=False, shared_weights=False),
        2,
        backend="cuda",
    )
    value.layout = layout
    value.set_metadata()
    return value


def joint(function, x, weight, types):
    output = function(x, weight, types)
    energy = output.sin().sum() / (x.shape[0] + 1)
    force = -torch.autograd.grad(energy, x, create_graph=True)[0]
    loss = energy.square() + (force - 0.01).square().sum()
    return output, force, loss, torch.autograd.grad(loss, weight)[0]


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("nodes", [0, 1, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_force_loss(layout, wide, nodes, dtype, monkeypatch):
    monkeypatch.setattr(config, "LINEAR_NATIVE", True)
    layer = module(layout, wide)
    x = torch.randn(layer.input_dim, nodes, device="cuda", dtype=dtype).T * 0.1
    w = (torch.randn(4, 2, 2 * layer.weight_numel, device="cuda", dtype=dtype) * 0.1)[
        ..., ::2
    ]
    types = (torch.arange(2 * nodes, device="cuda") % 4 - 4)[::2]
    x.requires_grad_()
    w.requires_grad_()
    actual = joint(layer, x, w, types)
    expected = joint(
        lambda *args: reference(layer, *args),
        x.detach().cpu().double().requires_grad_(),
        w.detach().cpu().double().requires_grad_(),
        types.cpu(),
    )
    atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (3e-12, 3e-11)
    torch.testing.assert_close(
        tuple(value.detach().cpu().double() for value in actual),
        expected,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_recursive_derivatives(layout, monkeypatch):
    monkeypatch.setattr(config, "LINEAR_NATIVE", True)
    layer = module(layout)
    x = torch.randn(5, layer.input_dim, device="cuda", dtype=torch.float64) * 0.1
    w = torch.randn(4, 2, layer.weight_numel, device="cuda", dtype=torch.float64) * 0.1
    types = torch.tensor([-1, -4, -2, -1, -3], device="cuda")
    values = []
    for function, device in ((layer, "cuda"), (lambda *a: reference(layer, *a), "cpu")):
        a, b = (value.to(device).detach().requires_grad_() for value in (x, w))
        value = function(a, b, types.to(device)).sin().square().sum()
        derivatives = []
        for order in range(1, 6):
            gradient = torch.autograd.grad(value, (a, b), create_graph=True)
            derivatives.append(tuple(part.detach().cpu() for part in gradient))
            value = sum(part.sin().sum() for part in gradient) / (1 + order)
        values.append(derivatives)
    torch.testing.assert_close(*values, atol=3e-12, rtol=3e-11)
    dx, dw = torch.randn_like(x) * 0.1, torch.randn_like(w) * 0.1
    results = []
    for function in (layer, lambda *args: reference(layer, *args)):
        gradient = torch.func.grad(
            lambda a, b: function(a, b, types).sin().square().sum(), argnums=(0, 1)
        )
        results.append(
            torch.vmap(lambda a, b: torch.func.jvp(gradient, (a, b), (dx, dw)))(
                torch.stack((x, x * 0.9)), torch.stack((w, w * 1.1))
            )
        )
    torch.testing.assert_close(*results, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_table_adjoint(layout, dtype, monkeypatch):
    monkeypatch.setattr(config, "LINEAR_NATIVE", True)
    ci, co, dim = 65, 67, 3
    metadata = repr(
        (ci * dim, co * dim, ci * co, 1, ((0, 0, 0, ci, co, dim, 3**-0.5),), layout)
    )
    x = torch.randn(3, ci * dim, dtype=dtype, device="cuda") * 0.1
    weight = torch.randn(1024, 1, ci * co, dtype=dtype, device="cuda") * 0.1
    output = torch.randn(3, co * dim, dtype=dtype, device="cuda") * 0.1
    types = torch.tensor([0, 1023, 17], device="cuda")
    assert weight.numel() * weight.element_size() > 16 << 20
    actual = contract(metadata, 0, types, x, weight, output).cpu().double()
    expected = contract_reference(
        metadata,
        0,
        types.cpu(),
        x.cpu().double(),
        weight.cpu().double(),
        output.cpu().double(),
    )
    atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (3e-12, 3e-11)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
