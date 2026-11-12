"""Torch expression kernels, reference autograd and recursive derivatives."""

import pytest
import torch

from equivariantx.ops import config, get_kernel
from equivariantx.ops.kernels.torch.edge import evaluate_recomputed as evaluate
from equivariantx.ops.kernels.torch.edge import evaluate_torch
from equivariantx.utils.program import Program


def program_case(device="cpu", edges=4):
    program = Program()
    x = program.input(0, "source", 2)
    radial = program.input(1, "edge", 2)
    matrix = program.input(2, "shared", 6)
    x = program.binary("add", x, radial)
    x = program.matmul(x, matrix, 1, 2, 3)
    x = program.unary("silu", x)
    x = program.add("normalize", 3, (x,), (0.5, True))
    selected = program.slice(x, 1, 2)
    metadata = repr((tuple(program.nodes), ((selected, 0, "target"), (x, 1, "shared"))))
    source = torch.tensor([0, 1, 2, 0][:edges], device=device, dtype=torch.long)
    target = torch.tensor([1, 2, 0, 2][:edges], device=device, dtype=torch.long)
    values = tuple(
        torch.randn(*shape, device=device, dtype=torch.float64).requires_grad_()
        for shape in ((3, 2), (edges, 2), (1, 6))
    )
    return metadata, source, target, values


@pytest.mark.parametrize("edges", [0, 4])
def test_forward_and_two_derivatives(edges):
    metadata, source, target, values = program_case(edges=edges)

    def function(*inputs):
        return tuple(evaluate(metadata, inputs, source, target, 3))

    torch.testing.assert_close(
        function(*values), tuple(evaluate_torch(metadata, values, source, target, 3))
    )
    assert torch.autograd.gradcheck(function, values, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, values, fast_mode=True)


def test_four_derivative_orders():
    metadata, source, target, values = program_case()
    optimized = evaluate(metadata, values, source, target, 3)
    reference = evaluate_torch(metadata, values, source, target, 3)
    actual = sum(value.square().sum() for value in optimized)
    expected = sum(value.square().sum() for value in reference)
    for order in range(4):
        actual_grad = torch.autograd.grad(actual, values, create_graph=True)
        expected_grad = torch.autograd.grad(expected, values, create_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-8)
        actual = sum(value.sum() for value in actual_grad)
        expected = sum(value.sum() for value in expected_grad)


def test_jvp_and_vmap():
    metadata, source, target, values = program_case()

    def optimized(*inputs):
        return tuple(evaluate(metadata, inputs, source, target, 3))

    def reference(*inputs):
        return tuple(evaluate_torch(metadata, inputs, source, target, 3))

    tangents = tuple(torch.randn_like(value) for value in values)
    torch.testing.assert_close(
        torch.func.jvp(optimized, values, tangents),
        torch.func.jvp(reference, values, tangents),
    )
    batched = tuple(torch.stack((value, value * 1.1)) for value in values)
    torch.testing.assert_close(
        torch.vmap(optimized)(*batched), torch.vmap(reference)(*batched)
    )


def test_functional_indices_do_not_escape(double_precision):
    from equivariantx.ops.kernels.torch.edge import program_indices

    program_indices.cache_clear()
    program = Program()
    x = program.input(0, "edge", 4)
    value = program.gather(x, (2, 0, 2))
    value = program.scatter(value, (1, 3, 0), 4)
    metadata = repr((tuple(program.nodes), ((value, 0, "edge"),)))
    index = torch.arange(3)
    x, direction = torch.randn(3, 4), torch.randn(3, 4)

    def function(x):
        return evaluate_torch(metadata, (x,), index, index, 3)[0]

    def reference(x):
        return x.new_zeros(x.shape).index_add(
            1, torch.tensor([1, 3, 0]), x[:, [2, 0, 2]]
        )

    for scale in (1.0, 0.7, 1.0):
        torch.testing.assert_close(
            torch.func.jvp(function, (x * scale,), (direction,)),
            torch.func.jvp(reference, (x * scale,), (direction,)),
        )
        torch.testing.assert_close(function(x * scale), reference(x * scale))
    program_indices.cache_clear()


def test_reference_switch(monkeypatch):
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", True)
    metadata, source, target, values = program_case()
    results = []
    for reference in (False, True, False):
        monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
        result = get_kernel("edge_program")(metadata, values, source, target, 3)
        results.append(result)
        assert (
            type(result[0].grad_fn).__name__ == "EdgeProgramBackward"
        ) is not reference
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])


def test_recompute_switch(monkeypatch):
    metadata, source, target, values = program_case()
    results = []
    for recompute in (False, True, False):
        monkeypatch.setattr(config, "TORCH_RECOMPUTE", recompute)
        result = get_kernel("edge_program")(metadata, values, source, target, 3)
        results.append(result)
        assert (type(result[0].grad_fn).__name__ == "EdgeProgramBackward") is recompute
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])


def test_unused_operands_have_zero_gradients():
    metadata, source, target, values = program_case()
    unused = torch.randn(3, dtype=torch.float64, requires_grad=True)
    for function in (evaluate, evaluate_torch):
        result = function(metadata, (*values, unused), source, target, 3)
        gradient = torch.autograd.grad(sum(value.sum() for value in result), unused)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(unused))


@pytest.mark.parametrize(
    "left_shared,right_shared",
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("edges", [0, 4])
def test_matrix_storage_and_reduction(left_shared, right_shared, reduce, edges):
    program = Program()
    left = program.input(0, "shared" if left_shared else "edge", 6)
    right = program.input(1, "shared" if right_shared else "edge", 12)
    result = program.matmul(left, right, 2, 3, 4)
    metadata = repr(
        (tuple(program.nodes), ((result, 0, "shared" if reduce else "edge"),))
    )
    index = torch.zeros(edges, dtype=torch.long)
    operands = tuple(
        torch.randn(
            1 if shared else edges, size, dtype=torch.float64, requires_grad=True
        )
        for shared, size in ((left_shared, 6), (right_shared, 12))
    )
    expected = operands[0].reshape(-1, 2, 3).expand(edges, 2, 3) @ operands[1].reshape(
        -1, 3, 4
    ).expand(edges, 3, 4)
    expected = expected.flatten(1)
    if reduce:
        expected = expected.sum(0, keepdim=True)
    actual = evaluate(metadata, operands, index, index, 1)[0]
    torch.testing.assert_close(actual, expected)
    for order in range(2):
        actual_grad = torch.autograd.grad(
            actual.sin().sum(), operands, create_graph=True
        )
        expected_grad = torch.autograd.grad(
            expected.sin().sum(), operands, create_graph=True
        )
        torch.testing.assert_close(actual_grad, expected_grad)
        actual = sum(value.sum() for value in actual_grad)
        expected = sum(value.sum() for value in expected_grad)


@pytest.mark.parametrize("width", [64, 128, 1024])
def test_radial_linear_force_gradient(width):
    program = Program()
    radial = program.input(0, "edge", width)
    matrix = program.input(1, "shared", width * 32)
    result = program.unary("silu", program.matmul(radial, matrix, 1, width, 32))
    metadata = repr((tuple(program.nodes), ((result, 0, "edge"),)))
    index = torch.arange(5)
    radial = torch.randn(5, width, dtype=torch.float64, requires_grad=True)
    matrix = (
        torch.randn(1, width * 32, dtype=torch.float64) / width**0.5
    ).requires_grad_()
    actual = evaluate(metadata, (radial, matrix), index, index, 5)[0]
    expected = torch.nn.functional.silu(radial @ matrix.reshape(width, 32))
    torch.testing.assert_close(actual, expected)
    forces = [
        torch.autograd.grad(value.square().sum(), radial, create_graph=True)[0]
        for value in (actual, expected)
    ]
    gradients = [
        torch.autograd.grad(force.square().sum(), matrix)[0] for force in forces
    ]
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA peak memory")
def test_shared_matrix_force_gradient_memory():
    width, channels, edges = 1024, 64, 256
    program = Program()
    radial = program.input(0, "edge", width)
    matrix = program.input(1, "shared", width * channels)
    result = program.unary("silu", program.matmul(radial, matrix, 1, width, channels))
    metadata = repr((tuple(program.nodes), ((result, 0, "edge"),)))
    index = torch.arange(edges, device="cuda")
    values = (
        torch.randn(
            edges, width, dtype=torch.float64, device="cuda", requires_grad=True
        ),
        torch.randn(
            1, width * channels, dtype=torch.float64, device="cuda", requires_grad=True
        ),
    )

    def force_gradient():
        prediction = evaluate(metadata, values, index, index, edges)[0]
        force = torch.autograd.grad(
            prediction.square().mean(), values[0], create_graph=True
        )[0]
        return torch.autograd.grad(force.square().sum(), values[1])[0]

    force_gradient()  # Initialize BLAS workspaces before measuring activations.
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    gradient = force_gradient()
    assert torch.isfinite(gradient).all()
    # A per-edge matrix outer product alone occupies 128 MiB for this shape.
    assert torch.cuda.max_memory_allocated() - allocated < 64 * 2**20


def test_compiler_sees_torch_operations():
    metadata, source, target, values = program_case()

    class Module(torch.nn.Module):
        def forward(self, x, radial, matrix):
            return tuple(evaluate(metadata, (x, radial, matrix), source, target, 3))

    compiled = torch.compile(Module(), backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*values), Module()(*values))
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(Module())(*values)
    assert all(not str(node.target).startswith("eqx.") for node in graph.graph.nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
def test_force_training_aoti_in_fresh_process(tmp_path):
    import os
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    metadata, source, target, values = program_case("cuda")

    def joint(x, radial, matrix, source, target):
        predictions = evaluate(metadata, (x, radial, matrix), source, target, 3)
        energy = predictions[0].square().sum() + predictions[1].sum()
        forces = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = (forces - 0.2).square().sum() + energy.square() * 0.1
        gradient = torch.autograd.grad(loss, matrix)[0]
        return loss, forces, gradient

    inputs = (*values, source, target)
    expected = joint(*inputs)
    graph = make_fx(joint)(*inputs)
    assert all(not str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(
        graph, tuple(value.detach() for value in inputs), strict=True
    )
    package_path = tmp_path / "joint.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package_path))
    data_path = tmp_path / "data.pt"
    torch.save(
        {
            "inputs": tuple(value.detach().cpu() for value in inputs),
            "expected": tuple(value.detach().cpu() for value in expected),
        },
        data_path,
    )
    script = """
import sys
import torch
data = torch.load(sys.argv[2], weights_only=True)
runner = torch._inductor.aoti_load_package(sys.argv[1])
with torch.no_grad():
    actual = runner(*(value.cuda() for value in data['inputs']))
for result, expected in zip(actual, data['expected']):
    torch.testing.assert_close(result.cpu(), expected, atol=1e-8, rtol=1e-8)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package_path), str(data_path)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
