"""Native and rematerialized edge programs under functional transformations."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.edge import edge_program, evaluate
from equivariantx.ops.common.torch.edge import evaluate_torch
from equivariantx.tests.ops.common.torch.test_edge import program_case
from equivariantx.utils.program import Program


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frame fusion")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [2, 32])
def test_frame_fusion_derivatives(dtype, channels, monkeypatch):
    from equivariantx.ops.common.cuda import execution, frame
    from equivariantx.utils.program import next_adjoint

    width = 3 * channels
    layout = repr(
        (
            width,
            width,
            9,
            (
                (
                    0,
                    channels,
                    3,
                    0,
                    ((0, 0, 1), (channels, 1, -1), (2 * channels, 2, 1)),
                ),
            ),
        )
    )
    program = Program()
    x = program.input(0, "source", width)
    y = program.input(0, "target", width)
    rotation = program.input(1, "edge", 9)
    weights = program.input(2, "edge", width)
    empty = program.constant(width, 0)
    left = program.add("frame", width, (x, rotation, empty), (layout, 2))
    right = program.add("frame", width, (y, rotation, empty), (layout, 2))
    local = program.binary("mul", program.binary("add", left, right), weights)
    output = program.add("frame", width, (empty, rotation, local), (layout, 0))
    metadata = repr((tuple(program.nodes), ((output, 0, "target"),)))
    backward = next_adjoint(metadata, (0, 1, 2), (0,), 3)
    assert execution.execution_plan(backward)[-2]

    contract = frame.contract

    def matrix_only(*args, **kwargs):
        assert args[6] == 1, "Rotations must remain inside their consuming expression"
        return contract(*args, **kwargs)

    monkeypatch.setattr(frame, "contract", matrix_only)
    source = torch.tensor([0, 2, 1, 2, 0, 1, 0], device="cuda")
    target = torch.tensor([1, 0, 2, 2, 1, 1, 2], device="cuda")
    values = tuple(
        (torch.randn(shape, device="cuda", dtype=dtype) * 0.2).requires_grad_()
        for shape in ((4, width), (7, 9), (7, width))
    )
    actual = edge_program("cuda", metadata, 4, source, target, *values, inline=False)[0]
    expected = evaluate_torch(metadata, values, source, target, 4)[0]
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        gradients = [
            torch.autograd.grad(value.sin().sum(), values, create_graph=True)
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([x.flatten() for x in gradient]) * 0.1 for gradient in gradients
        ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


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
@pytest.mark.parametrize("node_projection", [False, True])
def test_native_four_derivatives(edges, node_projection):
    metadata, source, target, values = program_case("cuda", edges)
    if node_projection:
        program = Program()
        x = program.input(0, "source", 2)
        radial = program.input(1, "edge", 2)
        weight = program.input(2, "shared", 6)
        local = program.matmul(x, weight, 1, 2, 3)
        scale = program.gather(radial, (0, 1, 0))
        output = program.binary("mul", program.unary("silu", local), scale)
        metadata = repr((tuple(program.nodes), ((output, 0, "target"),)))
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
def test_default_force_gradient_aoti_updates(tmp_path):
    metadata, source, target, values = program_case("cuda")

    def joint(native, *inputs):
        outputs = (
            evaluate(metadata, inputs, source, target, 3)
            if native
            else evaluate_torch(metadata, inputs, source, target, 3)
        )
        energy = 1 + sum(value.sin().mean() for value in outputs)
        force = -torch.autograd.grad(energy, inputs[0], create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, inputs[-1])[0]

    graph = make_fx(lambda *args: joint(True, *args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(graph, tuple(value.detach() for value in values))
    package = tmp_path / "edge_force.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    rows = []
    x, radial, weight = values
    for _ in range(3):
        expected = joint(False, x, radial, weight)
        rows.append(
            dict(
                inputs=tuple(value.detach().cpu() for value in (x, radial, weight)),
                expected=tuple(value.detach().cpu() for value in expected),
            )
        )
        weight = (weight - 0.001 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "cases.pt"
    torch.save(rows, data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    actual=run(*(value.cuda() for value in row['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),row['expected'],atol=2e-9,rtol=2e-9)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
