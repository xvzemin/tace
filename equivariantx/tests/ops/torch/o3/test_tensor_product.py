"""Factorized products, radial fusion and joint force-training graphs."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import conv, o3
from equivariantx.ops import config
from equivariantx.tests.conv.testing import assert_native_autograd


def tensor_product(kind, backend="torch", device="cpu", degree=2):
    cls = o3.GauntTensorProduct if kind == "gaunt" else o3.MatrixTensorProduct
    irreps = "+".join(f"2x{l}{'o' if l % 2 else 'e'}" for l in range(degree + 1))
    edge = "+".join(f"{l}{'o' if l % 2 else 'e'}" for l in range(degree + 1))
    return cls(
        irreps,
        edge,
        irreps,
        shared_weights=False,
        backend=backend,
    ).to(device=device, dtype=torch.float64)


@pytest.mark.parametrize("kind", ["gaunt", "matrix"])
def test_independent_reference_and_high_orders(kind, monkeypatch):
    module = tensor_product(kind, degree=1 if kind == "gaunt" else 2)
    values = tuple(
        (torch.randn(*shape, dtype=torch.float64) * 0.1).requires_grad_()
        for shape in (
            (2, 1, module.input_dim1),
            (1, 3, module.input_dim2),
            (2, 3, module.weight_numel),
        )
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = module(*values)
    assert_native_autograd(expected)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    graph = make_fx(module)(*values)
    actual = graph(*values)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(5):
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        pairs = [
            torch.autograd.grad((value.sin() * probe).sum(), values, create_graph=True)
            for value in (actual, expected)
        ]
        actual, expected = (torch.cat([v.flatten() for v in pair]) for pair in pairs)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    tangents = tuple(torch.randn_like(value) for value in values)
    ordinary = torch.func.jvp(module, values, tangents)
    with torch.autograd.forward_ad.dual_level():
        dual = tuple(
            torch.autograd.forward_ad.make_dual(x, v) for x, v in zip(values, tangents)
        )
        result = torch.autograd.forward_ad.unpack_dual(module(*dual))
    torch.testing.assert_close(result, ordinary)


@pytest.mark.parametrize("kind", ["gaunt", "matrix"])
@pytest.mark.parametrize("mode", ["last", "full"])
@pytest.mark.parametrize("width", [64, 128, 1024])
def test_radial_modes_and_mixed_gradients(kind, mode, width, monkeypatch):
    module = tensor_product(kind)
    cls = (
        conv.GauntTensorProductConv if kind == "gaunt" else conv.MatrixTensorProductConv
    )
    layer = cls(module, radial_mode=mode)
    network = torch.nn.Sequential(
        torch.nn.Linear(3, width),
        torch.nn.SiLU(),
        torch.nn.Linear(width, module.weight_numel),
    ).double()
    x = (torch.randn(3, module.input_dim1, dtype=torch.float64) * 0.1).requires_grad_()
    y = (torch.randn(5, module.input_dim2, dtype=torch.float64) * 0.1).requires_grad_()
    radial = (torch.randn(5, 3, dtype=torch.float64) * 0.1).requires_grad_()
    edges = torch.tensor([[0, 1, 2, 1, 2], [1, 2, 0, 0, 1]])
    values = (x, y, radial, *network.parameters())
    results = []
    for reference in (True, False):
        monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
        output = layer(x, y, edges, radial=radial, radial_network=network)
        if reference:
            assert_native_autograd(output)
        forces = torch.autograd.grad(output.sin().sum(), (y, radial), create_graph=True)
        loss = sum(force.square().sum() for force in forces) + output.square().mean()
        results.append((output, *forces, *torch.autograd.grad(loss, values)))
    torch.testing.assert_close(results[0], results[1], atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("mode", ["last", "full"])
@pytest.mark.parametrize("edges_count", [0, 5])
def test_convolution_broadcast_factors_and_partitioned_radial(
    mode, edges_count, monkeypatch
):
    module = tensor_product("gaunt")
    layer = conv.GauntTensorProductConv(module, radial_mode=mode)
    x = torch.randn(4, module.input_dim1, dtype=torch.float64, requires_grad=True)
    y = torch.randn(
        edges_count, module.input_dim2, dtype=torch.float64, requires_grad=True
    )
    w = torch.randn(module.weight_numel, dtype=torch.float64, requires_grad=True)
    edges = torch.randint(4, (2, edges_count))
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    graph = make_fx(lambda x, y, w: layer(x, y, edges, w))
    actual = graph(x, y, w)(x, y, w)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = layer(x, y, edges, w)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), (x, y, w)),
        torch.autograd.grad(expected.square().sum(), (x, y, w)),
        atol=3e-10,
        rtol=3e-10,
    )
    network = torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.SiLU(), torch.nn.Linear(8, module.weight_numel)
    ).double()
    source = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    shared = torch.randn(1, 1, dtype=torch.float64, requires_grad=True)
    radial = ((source, "source"), (target, "target"), (shared, "shared"))
    expected = layer(x, y, edges, radial=radial, radial_network=network)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = make_fx(
        lambda x, y: layer(x, y, edges, radial=radial, radial_network=network)
    )(x, y)(x, y)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)


def training_problem(kind, backend, device):
    module = tensor_product(kind, backend, device)
    values = tuple(
        (torch.randn(3, dim, dtype=torch.float64, device=device) * 0.2).requires_grad_()
        for dim in (module.input_dim1, module.input_dim2, module.weight_numel)
    )

    def operation(x, y, weights):
        output = module(x, y, weights)
        energy = output.sin().sum()
        force = -torch.autograd.grad(energy, y, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return output, force, loss, torch.autograd.grad(loss, weights)[0]

    return operation, values


def convolution_training_problem(
    mode, width, device="cuda", backend="cuda", *, dynamic=False
):
    class RadialProblem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            product = o3.GauntTensorProduct(
                "2x0e+2x1o",
                "0e+1o",
                "2x0e+2x1o",
                shared_weights=False,
                backend=backend,
            )
            self.layer = conv.GauntTensorProductConv(product, radial_mode=mode)
            self.radial = torch.nn.Sequential(
                torch.nn.Linear(3, width),
                torch.nn.SiLU(),
                torch.nn.Linear(width, product.weight_numel),
            )
            self.register_buffer("edges", torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]]))

        def forward(self, x, y, radial):
            return self.layer(
                x, y, self.edges, radial=radial, radial_network=self.radial
            )

    module = RadialProblem().to(device=device, dtype=torch.float64)
    values = tuple(
        (torch.randn(*shape, dtype=torch.float64, device=device) * 0.1).requires_grad_()
        for shape in ((3, 8), (4, 4), (4, 3))
    ) + tuple(module.parameters())
    names = tuple(dict(module.named_parameters()))
    if dynamic:
        values = (
            *values[:3],
            module.edges,
            *(value.detach().requires_grad_() for value in values[3:]),
        )

    def operation(x, y, radial, *parameters):
        edges, parameters = (
            (parameters[0], parameters[1:]) if dynamic else (None, parameters)
        )
        state = dict(zip(names, parameters))
        if dynamic:
            state["edges"] = edges
        output = torch.func.functional_call(module, state, (x, y, radial))
        energy = output.sin().sum()
        forces = torch.autograd.grad(energy, (y, radial), create_graph=True)
        loss = energy.square() + sum(force.square().sum() for force in forces)
        return output, *forces, loss, *torch.autograd.grad(loss, parameters)

    return operation, values


@pytest.mark.parametrize("kind", ["gaunt", "matrix"])
def test_joint_compilation_parameter_updates(kind, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    operation, values = training_problem(kind, "torch", "cpu")
    graph = make_fx(operation)(*values)
    compiled = torch.compile(graph, fullgraph=True)
    for _ in range(3):
        actual = compiled(*values)
        monkeypatch.setattr(config, "TORCH_REFERENCE", True)
        expected = operation(*values)
        monkeypatch.setattr(config, "TORCH_REFERENCE", False)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        values = (
            *values[:2],
            (values[2] - 0.01 * expected[-1]).detach().requires_grad_(),
        )


def test_full_capture_recomputes_only_radial(monkeypatch):
    from equivariantx.ops.common.torch import edge
    from equivariantx.utils.metadata import parse_metadata

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    operation, values = convolution_training_problem("full", 8, "cpu", "torch")
    programs = []
    evaluate = edge.evaluate_recomputed

    def record(metadata, *args):
        programs.append(parse_metadata(metadata)[0])
        return evaluate(metadata, *args)

    monkeypatch.setattr(edge, "evaluate_recomputed", record)
    graph = make_fx(operation)(*values)
    assert len(programs) == 1
    assert all(node[0] != "product" for node in programs[0])
    actual = graph(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    torch.testing.assert_close(actual, operation(*values), atol=3e-11, rtol=3e-11)
