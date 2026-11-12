"""Radial policies, sparse paths and mixed parameter derivatives."""

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.conv import O3TensorProductConv
from equivariantx.ops import config
from equivariantx.ops.convolution import convolve_torch


def convolution_case(width=4, edges=7, shared=False, device="cpu"):
    tp = o3.TensorProduct(
        "2x0e+2x1o",
        "0e+2x1o",
        "2x0e+2x1o+2x1o",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 2, "uvu", False),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    module = O3TensorProductConv(tp).double().to(device)
    index = torch.randint(3, (2, edges), device=device)
    inputs = tuple(
        torch.randn(*shape, dtype=torch.float64, device=device).requires_grad_()
        for shape in (
            (3, tp.irreps_in1.dim),
            (1 if shared else edges, tp.irreps_in2.dim),
            (1 if shared else edges, width),
            (width, tp.weight_numel),
        )
    )
    return module, inputs, index


@pytest.mark.parametrize("edges", [0, 7])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_convolution_derivatives(edges, shared, recompute, double_precision):
    module, inputs, index = convolution_case(edges=edges, shared=shared)

    def optimized(*values):
        return convolve_torch(
            module.kernel_metadata,
            *values,
            index,
            3,
            module.output_dim,
            recompute=recompute,
        )

    actual = optimized(*inputs)
    expected = module.reference(*inputs, index, 3)
    torch.testing.assert_close(actual, expected)
    actual, expected = actual.sin().mean(), expected.sin().mean()
    directions = tuple(
        torch.randn_like(value) / max(1, value.numel()) for value in inputs
    )
    for _ in range(4):
        gradients = [
            torch.autograd.grad(value, inputs, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
        actual, expected = [
            sum((g * direction).sum() for g, direction in zip(grads, directions))
            for grads in gradients
        ]
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(optimized, inputs, tangents),
        torch.func.jvp(
            lambda *values: module.reference(*values, index, 3), inputs, tangents
        ),
    )


@pytest.mark.parametrize("width", [64, 128, 1024])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_full_radial_force_training(
    width, shared, backend, monkeypatch, double_precision
):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    module, (x, attrs, radial, _), index = convolution_case(
        width=3,
        shared=shared,
        device=device,
    )
    module.backend = backend
    module.radial_mode = "full"
    network = torch.nn.Sequential(torch.nn.Linear(3, width), torch.nn.SiLU()).to(device)
    projection = torch.randn(width + 1, module.weight_numel, device=device) / width**0.5
    projection.requires_grad_()
    actual = module(x, attrs, radial, projection, index, radial_network=network)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module.backend = "torch"
    expected = module(x, attrs, radial, projection, index, radial_network=network)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    variables = (x, attrs, radial, projection, *network.parameters())
    actual, expected = actual.square().mean(), expected.square().mean()
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value, variables, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
        actual, expected = [
            sum(g.square().mean() for g in grads) / len(grads) for grads in gradients
        ]


def test_partitioned_radial_and_compilation(double_precision):
    module, (x, attrs, radial, projection), index = convolution_case(width=4)
    module.radial_mode = "full"
    network = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.SiLU()).double()
    source = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(3, 2, requires_grad=True)

    def function(x, attrs, radial, projection, source, target):
        partitions = ((radial, "edge"), (source, "source"), (target, "target"))
        return module(x, attrs, partitions, projection, index, radial_network=network)

    inputs = x, attrs, radial, projection, source, target
    expected = module.reference(
        x,
        attrs,
        network(torch.cat((radial, source[index[0]], target[index[1]]), -1)),
        projection,
        index,
        3,
    )
    actual = function(*inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), inputs, create_graph=True),
        torch.autograd.grad(expected.square().sum(), inputs, create_graph=True),
    )
    graph = make_fx(function)(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    compiled = torch.compile(graph, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs), expected)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_bipartite_source_gradient(backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    module, inputs, index = convolution_case(device=device)
    module.backend, module.radial_mode = backend, "full"
    index[1] = torch.arange(index.shape[1], device=device) % 5
    actual = module(*inputs, index, 5, radial_network=torch.nn.Sequential())
    expected = module.reference(*inputs, index, 5)
    torch.testing.assert_close(actual, expected)
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value.square().mean(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [
            sum(g.square().mean() for g in grads) / 100 for grads in gradients
        ]


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("radial_mode", ["last", "full"])
def test_dynamic_radial_program(backend, radial_mode, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.compiler.reset()
    device = "cuda" if backend == "cuda" else "cpu"
    module, _, _ = convolution_case(device=device)
    module.backend, module.radial_mode = backend, radial_mode
    network = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.SiLU()).to(device)

    def function(x, attrs, radial, projection, index):
        return module(x, attrs, radial, projection, index, radial_network=network)

    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(function, backend=compiler, dynamic=True, fullgraph=True)
    projection = torch.randn(8, module.weight_numel, device=device, requires_grad=True)
    for edges in (7, 11, 9):
        _, (x, attrs, radial, _), index = convolution_case(
            width=3, edges=edges, device=device
        )
        # Edge counts must stay independent of coincidentally equal model widths.
        for value, dimension in ((attrs, 0), (radial, 0), (index, 1)):
            torch._dynamo.mark_dynamic(value, dimension, min=2)
        actual = compiled(x, attrs, radial, projection, index)
        expected = function(x, attrs, radial, projection, index)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().mean(), projection)[0],
            torch.autograd.grad(expected.square().mean(), projection)[0],
        )
    assert len(graphs) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
@pytest.mark.parametrize("radial_mode", ["last", "full"])
def test_force_gradient_aoti(radial_mode, tmp_path, double_precision):
    import os
    import subprocess
    import sys

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, _, _ = convolution_case(device="cuda")
            self.conv.radial_mode = radial_mode
            self.network = torch.nn.Sequential(
                torch.nn.Linear(3, 8), torch.nn.SiLU()
            ).cuda()

        def forward(self, x, attrs, radial, projection, index):
            return self.conv(
                x, attrs, radial, projection, index, radial_network=self.network
            )

    model = Model()
    _, (x, attrs, radial, _), index = convolution_case(width=3, device="cuda")
    projection = torch.randn(
        8, model.conv.weight_numel, device="cuda", requires_grad=True
    )
    names, parameters = zip(*model.named_parameters())

    def joint(x, attrs, radial, projection, index, *parameters):
        prediction = torch.func.functional_call(
            model, dict(zip(names, parameters)), (x, attrs, radial, projection, index)
        )
        energy = prediction.square().mean()
        forces = -torch.autograd.grad(energy, radial, create_graph=True)[0]
        loss = (forces - 0.1).square().mean() + energy
        gradients = torch.autograd.grad(loss, (projection, *parameters))
        return loss, forces, *gradients

    inputs = (x, attrs, radial, projection, index, *parameters)
    graph = make_fx(joint)(*inputs)
    assert all(not str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    detached = tuple(value.detach() for value in inputs)
    exported = torch.export.export(graph, detached, strict=True)
    package = tmp_path / "convolution.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    # A packaged training graph must accept updated weights without recompiling.
    cases = []
    for scale in (1.0, 0.8):
        values = tuple(
            (
                value.detach() * scale if value.is_floating_point() else value.detach()
            ).requires_grad_(value.requires_grad)
            for value in inputs
        )
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in values),
                "expected": tuple(value.detach().cpu() for value in joint(*values)),
            }
        )
    data = tmp_path / "cases.pt"
    torch.save(cases, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = runner(*(value.cuda() for value in case['inputs']))
    for value, expected in zip(actual, case['expected']):
        torch.testing.assert_close(value.cpu(), expected, atol=1e-8, rtol=1e-8)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
