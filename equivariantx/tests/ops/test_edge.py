"""Native and rematerialized edge programs under functional transformations."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.decompositions import torch_decompositions
from equivariantx.ops.edge import edge_program, evaluate
from equivariantx.ops.kernels.torch.edge import evaluate_torch
from equivariantx.tests.ops.kernels.torch.test_edge import program_case
from equivariantx.utils.program import Program


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("edges", [0, 4])
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_bipartite_selected_gradient_jvp(backend, edges, compiler):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    program = Program()
    x = program.input(0, "source", 2)
    a = program.input(1, "edge", 2)
    root = program.binary("mul", x, a)
    metadata = repr((tuple(program.nodes), ((root, 0, "target"),)))
    source = torch.tensor([0, 1, 2, 0], device=device)[:edges]
    target = torch.tensor([0, 1, 3, 4], device=device)[:edges]
    values = tuple(
        torch.randn(shape, device=device, dtype=torch.float64)
        for shape in ((3, 2), (edges, 2))
    )
    tangents = tuple(torch.randn_like(value) for value in values)

    def function(x, a):
        return edge_program(backend, metadata, 5, source, target, x, a)[0].sum()

    def reference(x, a):
        return evaluate_torch(metadata, (x, a), source, target, 5)[0].sum()

    def transform(function, *values):
        return torch.func.jvp(torch.func.grad(function, argnums=0), values, tangents)

    expected = transform(reference, *values)

    def run(*values):
        return transform(function, *values)

    torch.compiler.reset()
    if compiler:
        run = torch.compile(run, backend=compiler, fullgraph=True)
    result = run(*values)
    assert all(value.shape == values[0].shape for value in result)
    torch.testing.assert_close(result, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA edge programs")
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_compiled_functional_derivatives(backend, compiler):
    metadata, source, target, values = program_case("cuda")
    tangents = tuple(torch.randn_like(value) for value in values)

    def function(*inputs):
        return edge_program(backend, metadata, 3, source, target, *inputs)

    def reference(*inputs):
        return tuple(evaluate_torch(metadata, inputs, source, target, 3))

    def transformed(function, *inputs):
        gradient = torch.func.grad(
            lambda *values: sum(value.sin().sum() for value in function(*values)),
            argnums=(0, 1, 2),
        )
        return (
            torch.func.jvp(function, inputs, tangents),
            torch.func.jvp(gradient, inputs, tangents),
            torch.func.jvp(
                lambda *values: torch.func.jvp(function, values, tangents)[1],
                inputs,
                tangents,
            ),
        )

    expected = transformed(reference, *values)
    torch.compiler.reset()

    def run(*inputs):
        return transformed(function, *inputs)

    if compiler:
        run = torch.compile(run, backend=compiler, fullgraph=True)
    torch.testing.assert_close(run(*values), expected, atol=2e-9, rtol=2e-9)

    def indexed(source, target, *inputs):
        return edge_program(backend, metadata, 3, source, target, *inputs)

    def indexed_reference(source, target, *inputs):
        return tuple(evaluate_torch(metadata, inputs, source, target, 3))

    batch = (
        torch.stack((source, source.roll(1))),
        torch.stack((target, target.roll(1))),
        *(torch.stack((value, value * 0.7)) for value in values),
    )
    torch.testing.assert_close(
        torch.vmap(indexed)(*batch),
        torch.vmap(indexed_reference)(*batch),
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA edge programs")
@pytest.mark.parametrize("edges", [0, 4])
def test_native_four_derivatives(edges):
    metadata, source, target, values = program_case("cuda", edges)
    actual = evaluate(metadata, values, source, target, 3)
    expected = evaluate_torch(metadata, values, source, target, 3)
    actual, expected = [
        sum(value.sin().sum() for value in outputs) for outputs in (actual, expected)
    ]
    for _ in range(4):
        actual_grad = torch.autograd.grad(actual, values, create_graph=True)
        expected_grad = torch.autograd.grad(expected, values, create_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-8)
        actual = sum(value.sum() for value in actual_grad)
        expected = sum(value.sum() for value in expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
def test_gradient_jvp_aoti(tmp_path):
    metadata, source, target, values = program_case("cuda")
    directions = tuple(torch.randn_like(value) for value in values)

    def transformed(backend, source, target, *inputs):
        def loss(*values):
            if backend == "cuda":
                outputs = evaluate(metadata, values, source, target, 3)
            else:
                outputs = evaluate_torch(metadata, values, source, target, 3)
            return sum(value.sin().sum() for value in outputs)

        gradient = torch.func.grad(loss, argnums=(0, 1, 2))
        primal, tangent = torch.func.jvp(gradient, inputs[:3], inputs[3:])
        return *primal, *tangent

    inputs = (source, target, *values, *directions)
    graph = make_fx(
        lambda *values: transformed("cuda", *values),
        decomposition_table=torch_decompositions(),
    )(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, inputs, strict=True)
    assert all(
        not str(node.target).startswith("eqx.")
        for node in exported.graph_module.graph.nodes
    )
    package = tmp_path / "edge_gradient_jvp.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.8):
        changed = tuple(
            value * scale if value.is_floating_point() else value.roll(1)
            for value in inputs
        )
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in changed),
                "expected": tuple(
                    value.detach().cpu() for value in transformed("torch", *changed)
                ),
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
        torch.testing.assert_close(value.cpu(), expected, atol=1e-9, rtol=1e-9)
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
