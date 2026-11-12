"""Native CUDA schedules and standalone compiler-visible primal programs."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.mbe.ace import SymmetricContraction
from equivariantx.ops.decompositions import symmetric_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decomposition")
def test_cuda_primal_aoti_without_eqx(tmp_path):
    module = (
        SymmetricContraction.from_entries(
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
    assert any(
        node.target == torch.ops.eqx.symmetric_contract.default
        for node in graph.graph.nodes
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
def test_cuda_adjoint_retains_native_lowering():
    module = (
        SymmetricContraction.from_entries(
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
