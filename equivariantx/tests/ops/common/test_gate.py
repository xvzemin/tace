"""Standalone gates must fuse into joint training and standalone AOTI graphs."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2, co3, o2, o3
from equivariantx.ops import config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA gates")


def gate_case(space, backend):
    namespace = {"o2": o2, "co2": co2, "o3": o3, "co3": co3}[space]
    gated = "2x1m+2x2m" if space in ("o2", "co2") else "2x1o+2x2e"
    return (
        namespace.Gate(
            "2x0e",
            [torch.nn.functional.silu],
            "4x0e",
            [torch.sigmoid],
            gated,
            backend=backend,
        )
        .double()
        .cuda()
    )


@pytest.mark.parametrize("space", ["o2", "co2", "o3", "co3"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_inline_joint_aoti(tmp_path, monkeypatch, space, backend):
    layer, reference = gate_case(space, backend), gate_case(space, "torch")
    value = torch.randn(
        5, layer.irreps_in.dim, dtype=torch.float64, device="cuda", requires_grad=True
    )
    weight = torch.randn(
        layer.irreps_in.dim, dtype=torch.float64, device="cuda", requires_grad=True
    )

    def operation(layer):
        def joint(value, weight):
            energy = layer(value * weight).sin().sum()
            force = -torch.autograd.grad(energy, value, create_graph=True)[0]
            loss = (force - 0.2).square().mean() + energy.square() * 0.01
            return loss, force, torch.autograd.grad(loss, weight)[0]

        return joint

    joint = operation(layer)
    with monkeypatch.context() as native_flags:
        native_flags.setattr(config, "GATE_INLINE", False)
        native_flags.setattr(config, "EDGE_PROGRAM_INLINE", False)
        native = make_fx(joint)(value, weight)
    if backend == "cuda":
        assert any(str(node.target).startswith("eqx.") for node in native.graph.nodes)
    monkeypatch.setattr(config, "GATE_INLINE", True)
    graph = make_fx(joint)(value, weight)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    expected = operation(reference)(value, weight)
    torch.testing.assert_close(graph(value, weight), expected, atol=2e-10, rtol=2e-10)
    exported = torch.export.export(
        graph, (value.detach(), weight.detach()), strict=True
    )
    package = tmp_path / "gate.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(reference)(value, weight)
        records.append(
            {
                "inputs": (value.detach().cpu(), weight.detach().cpu()),
                "outputs": tuple(x.detach().cpu() for x in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
        value = (value.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(x.cpu() for x in actual), record['outputs'], atol=2e-10, rtol=2e-10)
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


@pytest.mark.parametrize("space", ["o2", "co2", "o3", "co3"])
def test_inline_high_order(monkeypatch, space):
    layer, reference = gate_case(space, "cuda"), gate_case(space, "torch")
    value = (
        torch.randn(
            3,
            layer.irreps_in.dim,
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )
        * 0.2
    )
    tangent = torch.randn_like(value) * 0.1

    def derivatives(function, value):
        loss = function(value).sin().sum()
        outputs = []
        for _ in range(4):
            gradient = torch.autograd.grad(loss, value, create_graph=True)[0]
            outputs.append(gradient)
            loss = (gradient * tangent).sum()
        return tuple(outputs)

    monkeypatch.setattr(config, "GATE_INLINE", True)
    graph = make_fx(lambda x: derivatives(layer, x))(value)
    torch.testing.assert_close(
        graph(value), derivatives(reference, value), atol=2e-10, rtol=2e-10
    )
