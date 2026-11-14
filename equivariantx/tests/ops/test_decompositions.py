"""Native CUDA schedules and standalone compiler-visible primal programs."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.mbe.ace import UuuSymmetricContraction
from equivariantx.ops.decompositions import symmetric_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA edge lowering")
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("edges", [0, 7])
def test_edge_projection_joint(reference, edges, monkeypatch, double_precision):
    from equivariantx.ops import config
    from equivariantx.ops.common.edge import evaluate
    from equivariantx.ops.decompositions import torch_decompositions
    from equivariantx.utils.program import Program

    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    program = Program()
    x = program.input(0, "source", 4)
    radial = program.input(1, "edge", 3)
    weight = program.input(2, "shared", 12)
    coefficients = program.unary("tanh", program.matmul(radial, weight, 1, 3, 4))
    output = program.binary("mul", x, coefficients)
    metadata = repr((tuple(program.nodes), ((output, 0, "target"),)))
    source, target = (
        torch.arange(edges, device="cuda") % 3,
        torch.arange(edges, device="cuda") % 5,
    )
    values = tuple(
        torch.randn(shape, device="cuda").requires_grad_()
        for shape in ((3, 4), (edges, 3), (1, 12))
    )

    def joint(function, *inputs):
        result = function(*inputs)
        energy = result.square().mean()
        forces = -torch.autograd.grad(energy, inputs[1], create_graph=True)[0]
        loss = energy + (forces - 0.1).square().sum()
        return result, forces, *torch.autograd.grad(loss, (inputs[0], inputs[2]))

    def native(*inputs):
        return evaluate(metadata, inputs, source, target, 5)[0]

    def oracle(x, radial, weight):
        message = x[source] * (radial @ weight.reshape(3, 4)).tanh()
        return x.new_zeros((5, 4)).index_add(0, target, message)

    graph = make_fx(
        lambda *inputs: joint(native, *inputs),
        decomposition_table=torch_decompositions(),
    )(*values)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    if edges and not reference:
        assert any(
            node.target == torch.ops.aten.mm.default for node in graph.graph.nodes
        )
    run = torch.compile(graph, fullgraph=True)
    for scale in (1.0, 0.7):
        changed = tuple((x.detach() * scale).requires_grad_() for x in values)
        expected = joint(oracle, *changed)
        with torch.no_grad():
            actual = run(*(x.detach() for x in changed))
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decomposition")
@pytest.mark.parametrize("inline", [False, True])
def test_cuda_primal_aoti_without_eqx(inline, tmp_path, monkeypatch):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "POLYNOMIAL_INLINE", inline)
    module = (
        UuuSymmetricContraction.from_entries(
            "2x0e",
            "2x0e",
            [((0, 0), 0, 0, 1.0), ((0, 0, 0), 1, 0, -0.5)],
            2,
            backend="cuda",
        )
        .double()
        .cuda()
    )
    x = torch.randn(5, 2, 1, dtype=torch.float64, device="cuda")
    weight = torch.randn(2, 2, 2, dtype=torch.float64, device="cuda")
    species = torch.tensor([0, 1, 0, 0, 1], device="cuda")
    inputs = x, weight, species
    reference = module(*inputs)
    graph = make_fx(module)(*inputs)
    assert (
        any(
            node.target == torch.ops.eqx.symmetric_contract.default
            for node in graph.graph.nodes
        )
        is not inline
    )
    lowered = make_fx(
        module,
        decomposition_table={
            torch.ops.eqx.symmetric_contract.default: symmetric_forward
        },
    )(*inputs)
    assert all(
        node.target != torch.ops.eqx.symmetric_contract.default
        for node in lowered.graph.nodes
    )
    torch.testing.assert_close(lowered(*inputs), reference)
    exported = torch.export.export(graph, inputs, strict=True)
    path = tmp_path / "symmetric.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(path))
    data_path = tmp_path / "data.pt"
    torch.save(
        {"inputs": tuple(value.cpu() for value in inputs), "output": reference.cpu()},
        data_path,
    )
    script = """
import sys
import torch
data = torch.load(sys.argv[2], weights_only=True)
runner = torch._inductor.aoti_load_package(sys.argv[1])
with torch.no_grad():
    actual = runner(*(value.cuda() for value in data['inputs']))
torch.testing.assert_close(actual.cpu(), data['output'], atol=1e-10, rtol=1e-10)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path), str(data_path)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiled adjoints")
def test_cuda_adjoint_joint_compilation():
    module = (
        UuuSymmetricContraction.from_entries(
            "3x0e",
            "3x0e",
            [((0, 0), 0, 0, 0.4), ((0, 0, 0), 1, 0, -0.7)],
            2,
            backend="cuda",
        )
        .double()
        .cuda()
    )
    x = torch.randn(7, 3, 1, dtype=torch.float64, device="cuda", requires_grad=True)
    weight = torch.randn(
        2, 2, 3, dtype=torch.float64, device="cuda", requires_grad=True
    )
    species = torch.arange(7, device="cuda") % 2

    def joint(x, weight, species):
        energy = module(x, weight, species).square().sum()
        force = torch.autograd.grad(energy, x, create_graph=True)[0]
        gradient = torch.autograd.grad(force.square().sum(), weight)[0]
        return energy, force, gradient

    inputs = x, weight, species
    expected = joint(*inputs)
    graph = make_fx(joint)(*inputs)
    compiled = torch.compile(graph, fullgraph=True)
    with torch.no_grad():
        actual = compiled(*(value.detach() for value in inputs))
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
