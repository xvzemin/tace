"""Native and rematerialized edge programs under functional transformations."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.edge import edge_program, evaluate
from equivariantx.ops.common.torch.edge import evaluate_torch
from equivariantx.ops.compiler import compile_and_package
from equivariantx.tests.ops.common.torch.test_edge import program_case
from equivariantx.utils.program import Program


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph index lifetime")
def test_replay_pins_column_indices(monkeypatch):
    import weakref
    from collections import OrderedDict

    from equivariantx.ops.common.cuda import execution, graph

    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "1")
    monkeypatch.setattr(graph, "CONVOLUTION_GRAPH_MIN_CALLS", 1)
    monkeypatch.setattr(graph, "_GRAPHS", OrderedDict())
    x = torch.randn(9, 5, device="cuda")
    columns = (4, 1, 2)
    output = x.new_empty((9, 3))

    def run(inputs, outputs):
        outputs[0].copy_(execution.transform("gather", 3, inputs, columns))

    graph.execute("edge_program", "column_lifetime", run, (x,), (output,))
    pointer = weakref.ref(execution.indices(columns, x.device))
    execution.indices.cache_clear()
    assert pointer() is not None
    x.normal_()
    graph.execute("edge_program", "column_lifetime", run, (x,), (output,))
    torch.testing.assert_close(output, x[:, columns])


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_gate_higher_derivatives(backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if backend == "cuda" else "cpu"
    x = torch.tensor(
        [[-20.0, -0.4, 0.0, 0.6, 20.0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    index = torch.zeros(1, dtype=torch.long, device=device)
    program = Program()
    operand = program.input(0, "edge", x.shape[1])
    roots = tuple(
        (program.unary(name, operand), i, "edge")
        for i, name in enumerate(("silu", "sigmoid", "tanh"))
    )
    actual = edge_program(
        backend, repr((tuple(program.nodes), roots)), 1, index, index, x, inline=False
    )
    expected = (torch.nn.functional.silu(x), x.sigmoid(), x.tanh())
    for value, reference in zip(actual, expected, strict=True):
        for _ in range(5):
            (value,) = torch.autograd.grad(value.sum(), (x,), create_graph=True)
            (reference,) = torch.autograd.grad(reference.sum(), (x,), create_graph=True)
            torch.testing.assert_close(value, reference, atol=2e-13, rtol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay")
@pytest.mark.parametrize("shared_nodes", [False, True])
def test_frame_replay_changes_graph_size(shared_nodes, monkeypatch):
    from collections import OrderedDict

    from equivariantx.ops.common.cuda import graph
    from equivariantx.ops.common.cuda.edge import padding_layout

    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "1")
    monkeypatch.setattr(graph, "_GRAPHS", OrderedDict())
    monkeypatch.setattr(graph, "_CANDIDATES", OrderedDict())
    monkeypatch.setattr(graph, "CONVOLUTION_GRAPH_MIN_CALLS", 1)
    monkeypatch.setattr(graph, "MIN_EDGES", 4)
    monkeypatch.setattr(graph, "ROW_BUCKET", 8)
    p = Program()
    src = p.input(0, "source", 2)
    dst = p.input(0 if shared_nodes else 4, "target", 2)
    rot, w, b = p.input(1, "edge", 4), p.input(2, "shared", 4), p.input(3, "shared", 2)
    layout = repr((2, 2, 4, ((0, 1, 2, 0, ((0, 0, 1), (1, 1, 1))),)))
    local = p.add(
        "frame", 2, (p.binary("add", src, dst), rot, p.constant(2, 0)), (layout, 2)
    )
    y = p.unary("silu", p.binary("add", p.matmul(local, w, 1, 2, 2), b))
    meta = repr((tuple(p.nodes), ((y, 0, "target"),)))
    assert padding_layout(meta, indexed=True) is not None
    # A padded edge still has a nonzero bias. Discarding its private node is
    # safe, summing its output into a shared tensor would not be.
    assert (
        padding_layout(repr((tuple(p.nodes), ((y, 0, "shared"),))), indexed=True)
        is None
    )
    previous = None
    for nodes, edges in ((5, 12), (6, 10), (7, 13)):
        targets = nodes if shared_nodes else nodes - 2
        source = torch.randint(nodes, (edges,), device="cuda")
        target = torch.randint(targets, (edges,), device="cuda")
        shapes = [(nodes, 2), (edges, 4), (1, 4), (1, 2)]
        if not shared_nodes:
            shapes.append((targets, 2))
        values = [
            torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
            for shape in shapes
        ]
        actual = edge_program(
            "cuda", meta, targets, source, target, *values, inline=False
        )[0]
        reference = evaluate_torch(meta, values, source, target, targets)[0]

        def outputs(x):
            energy = x.square().sum()
            (forces,) = torch.autograd.grad(energy, values[1], create_graph=True)
            return (
                x,
                forces,
                *torch.autograd.grad(energy + forces.square().sum(), values),
            )

        for a, b in zip(outputs(actual), outputs(reference), strict=True):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-10)
        keys = set(graph._GRAPHS)
        assert keys
        if previous is not None:
            assert keys == previous
        previous = keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reductions")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_edge_dependent_projection_and_receiver_reduction(dtype):
    from equivariantx.ops.common.cuda.edge_codegen import launch

    N, E, width = 5, 103, 35
    index = torch.randint(N - 1, (2, E), device="cuda")
    x = torch.randn(N, 3, device="cuda", dtype=dtype)
    matrices = torch.randn(E, 3 * width, device="cuda", dtype=dtype)
    program = Program()
    node, matrix = program.input(0, "source", 3), program.input(1, "edge", 3 * width)
    root = program.matmul(node, matrix, 1, 3, width)
    metadata = repr((tuple(program.nodes), ((root, 0, "target"),)))
    actual = edge_program("cuda", metadata, N, *index, x, matrices, inline=False)[0]
    messages = torch.bmm(x[index[0], None, :], matrices.view(E, 3, width))[:, 0]
    expected = x.new_zeros(N, width).index_add_(0, index[1], messages)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    program = Program()
    root = program.input(0, "edge", width)
    for owner, ids in (("source", index[0]), ("target", index[1])):
        metadata = repr((tuple(program.nodes), ((root, 0, owner),)))
        output = x.new_ones(N, width)
        launch(metadata, [messages.contiguous()], *index, [output], mode="node")
        expected = x.new_ones(N, width).index_add_(0, ids, messages)
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_transpose_before_reinterpreted_matmul(backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    program = Program()
    x, w = program.input(0, "source", 3), program.input(1, "shared", 12)
    transposed = program.transpose(w, 2, 6)
    root = program.matmul(x, transposed, 1, 3, 4)
    metadata = repr((tuple(program.nodes), ((root, 0, "target"),)))
    index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device)
    x = torch.randn(3, 3, device=device, requires_grad=True)
    w = torch.randn(1, 12, device=device, requires_grad=True)
    actual = edge_program(backend, metadata, 3, *index, x, w, inline=False)[0]
    expected = evaluate_torch(metadata, (x, w), *index, 3)[0]
    for _ in range(3):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        y.sin().sum(), (x, w), create_graph=True
                    )
                ]
            )
            for y in (actual, expected)
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("primal_inline", [False, True])
def test_large_program_keeps_native_derivatives(dtype, primal_inline, monkeypatch):
    from equivariantx.ops import config
    from equivariantx.ops.common.compilation import edge_work

    program = Program()
    x = program.input(0, "source", 2)
    output = x
    for index in range(6 if primal_inline else 300):
        term = program.unary("sin", output) if primal_inline else x
        output = program.binary("add", output, program.scale(term, 1 / (index + 2)))
    metadata = repr((tuple(program.nodes), ((output, 0, "target"),)))
    if primal_inline:
        monkeypatch.setattr(config, "COMPILE_MAX_INLINE_WORK", edge_work(metadata))
    else:
        assert edge_work(metadata) > config.COMPILE_MAX_INLINE_WORK
    source = torch.tensor([0, 1, 2, 0], device="cuda")
    target = torch.tensor([1, 2, 0, 2], device="cuda")
    values = torch.randn(3, 2, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    if primal_inline:
        primal = make_fx(
            lambda features: edge_program("cuda", metadata, 3, source, target, features)
        )(values)
        assert "eqx.edge_program" not in primal.code

    def joint(features, backend="cuda"):
        output = edge_program(backend, metadata, 3, source, target, features)[0]
        force = torch.autograd.grad(output.sin().sum(), features, create_graph=True)[0]
        gradient = torch.autograd.grad(force.square().sum(), features)[0]
        return output, force, gradient

    graph = make_fx(lambda features: joint(features))(values)
    assert "eqx.edge_program" in graph.code
    assert len(list(graph.graph.nodes)) < 200
    if len(metadata) >= 4096:
        calls = [
            n for n in graph.graph.nodes if str(n.target) == "eqx.edge_program.default"
        ]
        assert calls and all(n.args[0].startswith("z:") for n in calls)
    compiled = torch.compile(graph, fullgraph=True)
    tolerance = 3e-4 if dtype == torch.float32 else 2e-10
    for scale in (1.0, 0.8):
        inputs = (values.detach() * scale).requires_grad_()
        with torch.no_grad():
            actual = compiled(inputs)
        torch.testing.assert_close(
            actual, joint(inputs, "torch"), atol=tolerance, rtol=tolerance
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frame fusion")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [2, 32])
def test_frame_fusion_derivatives(dtype, channels, monkeypatch):
    from equivariantx.ops.common.cuda import execution, frame
    from equivariantx.ops.common.cuda.edge_codegen import frame_rows
    from equivariantx.utils.metadata import parse_metadata
    from equivariantx.utils.program import next_adjoint

    if channels == 2:
        from e3nn import o3
        from equivariantx import o2

        irreps = [(channels, ir) for _, ir in o3.Irreps.spherical_harmonics(7)]
        irreps.insert(1, (4, o3.Irrep("1e")))
        layout = o2.LocalFrame(o3.Irreps(irreps), mmax=7).rotation_layout
        width, _, matrix_width, _ = parse_metadata(layout)
        assert all(frame_rows(layout, role) is not None for role in (0, 2))
    else:
        width, matrix_width = 3 * channels, 9
        rows = ((0, 0, 1), (channels, 1, -1), (2 * channels, 2, 1))
        layout = repr((width, width, matrix_width, ((0, channels, 3, 0, rows),)))
    program = Program()
    x = program.input(0, "source", width)
    y = program.input(0, "target", width)
    rotation = program.input(1, "edge", matrix_width)
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
        for shape in ((4, width), (7, matrix_width), (7, width))
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
    compile_and_package(exported, package_path=str(package))
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
    compile_and_package(exported, package_path=str(package))
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
