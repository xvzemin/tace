"""Native rectangular frame contractions in independent joint AOTI packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native Cartesian frames require CUDA"
)


def test_joint_aoti_without_eqx(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TENSOR_AXIS_INLINE", False)
    layers = {
        backend: co2.LocalFrame("2x1o+1x2e", backend=backend).double().cuda()
        for backend in ("torch", "cuda")
    }

    def operation(x, rotation, weight, ordinary=False):
        layer = layers["torch" if ordinary else "cuda"]
        y = layer(x * weight, rotation)
        energy = y.sin().sum() * 0.1
        force = -torch.autograd.grad(energy, rotation, create_graph=True)[0]
        loss = (force - 0.01).square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    x = (torch.randn(3, 15, device="cuda", dtype=torch.float64) * 0.2).requires_grad_()
    rotation = (
        torch.randn(3, 3, 3, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    weight = (
        torch.randn(15, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    values = x, rotation, weight
    graph = make_fx(lambda *args: operation(*args))(*values)
    assert any(
        "eqx.tensor_axis" in str(node.target)
        or "triton_kernel_wrapper" in str(node.target)
        for node in graph.graph.nodes
    )
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(v.detach() for v in values), strict=True
    )
    package = tmp_path / "local-frame.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(x, rotation, weight, True)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in (x, rotation, weight)),
                "output": tuple(v.detach().cpu() for v in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
        rotation = (rotation.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad(): actual=run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),record['output'],atol=3e-9,rtol=3e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
