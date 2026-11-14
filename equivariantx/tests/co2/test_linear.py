"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import co2, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("kind", ["linear", "uu"])
def test_forward_ad(kind, device):
    from equivariantx.tests.utils.testing import check_linear_forward_ad

    check_linear_forward_ad(co2, kind, device)


@pytest.mark.parametrize("path_mode", ["sum", "expand"])
def test_cartesian_uu_linear(double_precision, path_mode):
    a, b = "4x0e+4x0o+6x1m+2x3m", "2x0e+2x0o+4x1m+2x3m"
    reference = o2.UuLinear(a, b, 2, path_mode=path_mode).to(DEVICE)
    module = co2.UuLinear(a, b, 2, path_mode=path_mode).to(DEVICE)
    features = torch.randn(3, reference.irreps_in.dim, device=DEVICE)
    weight = torch.randn(3, reference.weight_numel, device=DEVICE)
    actual = co2.ChangeOfBasis(module.irreps_out, inverse=True).to(DEVICE)(
        module(co2.ChangeOfBasis(a).to(DEVICE)(features), weight)
    )
    torch.testing.assert_close(actual, reference(features, weight))


@pytest.mark.parametrize("path_normalization", ["element", "path"])
def test_linear_and_gate(double_precision, path_normalization):
    a, b = "2x0e+3x1mo+1x1mo+2x3m", "1x0e+2x1mo+2x3m+1x0o"
    ref = o2.Linear(
        a,
        b,
        biases=True,
        internal_weights=False,
        shared_weights=False,
        path_normalization=path_normalization,
    ).to(DEVICE)
    module = co2.Linear(
        a,
        b,
        biases=True,
        internal_weights=False,
        shared_weights=False,
        path_normalization=path_normalization,
        output_basis="circular",
    ).to(DEVICE)
    x = torch.randn(4, ref.irreps_in.dim, device=DEVICE)
    weight = torch.randn(1, ref.weight_numel, device=DEVICE)
    bias = torch.randn(4, ref.bias_numel, device=DEVICE)
    embed = co2.ChangeOfBasis(a).to(DEVICE)
    torch.testing.assert_close(module(embed(x), weight, bias), ref(x, weight, bias))
    args = (
        "2x0e+1x0oo",
        [torch.nn.functional.silu, torch.tanh],
        "2x0e+2x0oo",
        [torch.sigmoid, torch.tanh],
        "2x1mo+2x3m",
    )
    gate = co2.Gate(*args).to(DEVICE)
    state = gate.state_dict()
    state["mul.weight"] = torch.empty(0, device=DEVICE)
    gate.load_state_dict(state)
    compact_gate = o2.Gate(*args).to(DEVICE)
    x = compact_gate.irreps_in.randn(5, -1, device=DEVICE)
    actual = co2.ChangeOfBasis(gate.irreps_out, True).to(DEVICE)(
        gate(co2.ChangeOfBasis(gate.irreps_in).to(DEVICE)(x))
    )
    torch.testing.assert_close(actual, compact_gate(x))


def test_linear_compiled_derivatives(double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    torch.compiler.reset()
    module = co2.Linear("4x0e+2x1m", "4x0e+2x1m").to(DEVICE)

    def evaluate(x, weight):
        output = module(x, weight)
        gradient = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
        return output, gradient

    x = torch.randn(3, 8, device=DEVICE, requires_grad=True)
    weight = module.weight.detach().clone().requires_grad_()
    graph = make_fx(evaluate, tracing_mode="symbolic")(x, weight)
    compiled = torch.compile(graph, backend="inductor", fullgraph=True, dynamic=True)
    for size in (3, 5):
        x = torch.randn(size, 8, device=DEVICE, requires_grad=True)
        expected, actual = evaluate(x, weight), compiled(x, weight)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        gradients = [
            torch.autograd.grad(sum(y.square().sum() for y in values), (x, weight))
            for values in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("shared", [False, True])
def test_linear_sparse_paths(double_precision, shared):
    args = ("2x0e+3x1m+2x1m", "3x0e+2x1m+1x1m+1x2m")
    kwargs = dict(
        instructions=[(0, 0), (1, 1), (1, 1), (2, 1), (1, 2)],
        biases=True,
        internal_weights=False,
        shared_weights=shared,
    )
    reference = o2.Linear(*args, **kwargs).to(DEVICE)
    module = co2.Linear(*args, **kwargs).to(DEVICE)
    embed = co2.ChangeOfBasis(args[0]).to(DEVICE)
    output = co2.ChangeOfBasis(args[1]).to(DEVICE)
    x = torch.randn(3, reference.irreps_in.dim, device=DEVICE, requires_grad=True)
    weight = torch.randn(
        1 if shared else 3, module.weight_numel, device=DEVICE, requires_grad=True
    )
    bias = torch.randn(3, module.bias_numel, device=DEVICE, requires_grad=True)
    values = [output(reference(x, weight, bias)), module(embed(x), weight, bias)]
    for _ in range(3):
        torch.testing.assert_close(*values)
        values = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        value.sin().sum(), (x, weight, bias), create_graph=True
                    )
                ]
            )
            for value in values
        ]
