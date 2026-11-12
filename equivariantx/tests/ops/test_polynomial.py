"""Standalone compiled JVPs of native polynomial gradient programs."""

import os
import subprocess
import sys

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.mbe.ace import SymmetricContraction
from equivariantx.o3 import MoEElementLinear
from equivariantx.ops.decompositions import polynomial_decompositions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize("kind", ["symmetric", "linear"])
def test_gradient_jvp_aoti(kind, tmp_path, double_precision):
    if kind == "symmetric":
        modules = [
            SymmetricContraction.from_entries(
                "2x0e",
                "2x0e",
                [((0,), 0, 0, 0.4), ((0, 0, 0), 1, 0, -0.1)],
                2,
                backend=backend,
            ).cuda()
            for backend in ("cuda", "torch")
        ]
        shapes = ((5, 2, 1), (2, 2, 2))
    else:
        linear = o3.Linear(
            "2x0e+2x1o", "3x0e+3x1o", internal_weights=False, shared_weights=False
        )
        modules = [
            MoEElementLinear(linear, 2, backend).cuda() for backend in ("cuda", "torch")
        ]
        shapes = ((5, 16), (2, 2, linear.weight_numel))
    values = tuple(torch.randn(*shape, device="cuda") * 0.3 for shape in shapes)
    directions = tuple(torch.randn_like(value) for value in values)
    types = torch.tensor([0, 1, 0, 0, 1], device="cuda")
    inputs = *values, *directions, types

    def evaluate(module, x, w, dx, dw, types):
        gradient = torch.func.grad(
            lambda x, w: module(x, w, types).sin().sum(), argnums=(0, 1)
        )
        primal, tangent = torch.func.jvp(gradient, (x, w), (dx, dw))
        return *primal, *tangent

    graph = make_fx(
        lambda *values: evaluate(modules[0], *values),
        decomposition_table=polynomial_decompositions(),
    )(*inputs)
    # Forward AD emits unused boolean shape checks; export guards input shapes.
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, inputs, strict=True)
    assert all(
        not str(node.target).startswith("eqx.")
        for node in exported.graph_module.graph.nodes
    )
    package = tmp_path / "gradient_jvp.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.8):
        changed = tuple(
            value * scale if value.is_floating_point() else value for value in inputs
        )
        expected = evaluate(modules[1], *changed)
        cases.append(
            {
                "inputs": tuple(value.cpu() for value in changed),
                "expected": tuple(value.cpu() for value in expected),
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
