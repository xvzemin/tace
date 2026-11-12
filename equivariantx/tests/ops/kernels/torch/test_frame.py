"""Frame kernel transposes against native contractions."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import o2
from equivariantx.ops import config, frame, get_kernel, rotary, rotation
from equivariantx.ops.decompositions import torch_decompositions
from equivariantx.tests.ops.testing import assert_functional_derivatives
from equivariantx.utils.metadata import parse_metadata


def frame_case(kind, rows, role):
    if kind == "indexed_frame":
        module = o2.LocalFrame("2x0e + 2x1o + 1x1e + 1x2e", mmax=1)
        ni, no, nw, _ = parse_metadata(module.rotation_layout)
        values = [
            torch.randn(5, ni, device="cuda"),
            torch.randn(rows, nw, device="cuda"),
            torch.randn(rows, no, device="cuda"),
        ]
        index = torch.arange(rows, device="cuda") % 5
        parameters = index, 5, module.rotation_layout, role
        native = frame.contract
    elif kind == "frame_rotation":
        incoming, outgoing = (0, 1, 1, 1, 2, 2), (1, 1, 1, 0)
        values = [
            torch.randn(rows, len(incoming), 4, device="cuda"),
            torch.randn(rows, len(outgoing), len(incoming), device="cuda"),
            torch.randn(rows, len(outgoing), 4, device="cuda"),
        ]
        parameters = repr((incoming, outgoing)), role
        native = rotation.contract
    else:
        values = [
            torch.randn(rows, 14, 6, device="cuda"),
            torch.randn(rows, 14, 6, device="cuda"),
            torch.randn(rows, 3, 2, 2, device="cuda"),
            torch.randn(rows, 2, device="cuda"),
        ]
        parameters = 3, 2, 2, role
        native = rotary.contract
    return values, parameters, native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frame reference")
@pytest.mark.parametrize(
    "kind,role",
    [
        *(
            (kind, role)
            for kind in ("indexed_frame", "frame_rotation")
            for role in range(3)
        ),
        *(("rotary_product", role) for role in range(4)),
    ],
)
@pytest.mark.parametrize("rows", [0, 7])
def test_transposes(kind, role, rows, double_precision):
    values, parameters, native = frame_case(kind, rows, role)
    values = [value.mul(0.2).requires_grad_() for value in values]
    actual = native(*values, *parameters)
    expected = get_kernel(kind, "torch")(*values, *parameters)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    active = tuple(value for i, value in enumerate(values) if i != role)
    for _ in range(4):
        derivatives = [
            torch.autograd.grad(value.sin().sum(), active, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*derivatives, atol=1e-8, rtol=1e-8)
        actual, expected = [
            sum(gradient.sum() for gradient in gradients)
            / max(sum(gradient.numel() for gradient in gradients), 1)
            for gradients in derivatives
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frame transforms")
@pytest.mark.parametrize(
    "kind,role",
    [
        ("indexed_frame", 2),
        ("frame_rotation", 2),
        ("rotary_product", 3),
    ],
)
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_transforms(kind, role, compiler, double_precision):
    values, parameters, native = frame_case(kind, 7, role)
    values = tuple(value * 0.2 for value in values)
    dummy = torch.arange(7, device="cuda")

    def function(*inputs):
        return native(*inputs[:-1], *parameters)

    def reference(*inputs):
        return get_kernel(kind, "torch")(*inputs[:-1], *parameters)

    assert_functional_derivatives(function, reference, (*values, dummy), compiler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rotary capture")
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_rotary_capture_policy(inline, recompute, monkeypatch, double_precision):
    monkeypatch.setattr(config, "ROTARY_INLINE", inline)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", recompute)
    values, parameters, native = frame_case("rotary_product", 7, 3)
    values = tuple(value.requires_grad_() for value in values[:3])
    placeholder = values[0].new_empty((0, 2))

    def function(*inputs):
        prediction = native(*inputs, placeholder, *parameters)
        force = -torch.autograd.grad(
            prediction.square().sum(), inputs[-1], create_graph=True
        )[0]
        return force, *torch.autograd.grad(force.sin().sum(), inputs[:2])

    graph = make_fx(function)(*values)
    expanded = inline and not recompute
    assert (
        any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
        != expanded
    )
    expected = function(*values)
    compiled = torch.compile(graph, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(*values), expected, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frame AOTInductor")
@pytest.mark.parametrize(
    "kind,role",
    [
        ("indexed_frame", 2),
        ("frame_rotation", 2),
        ("rotary_product", 3),
    ],
)
def test_force_derivatives_aoti(kind, role, double_precision, tmp_path):
    values, parameters, native = frame_case(kind, 7, role)
    placeholder = values[role]
    inputs = tuple((value * 0.2).requires_grad_() for value in values[:role])

    def joint(function, *inputs):
        output = function(*inputs, placeholder, *parameters)
        energy = output.sin().sum() / max(output.numel(), 1)
        force = -torch.autograd.grad(energy, inputs[0], create_graph=True)[0]
        gradients = torch.autograd.grad(
            energy + (force - 0.01).square().sum(), inputs[1:]
        )
        return energy, force, *gradients

    graph = make_fx(
        lambda *inputs: joint(native, *inputs),
        decomposition_table=torch_decompositions(),
    )(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, inputs, strict=True)
    assert all(
        not str(node.target).startswith("eqx.")
        for node in exported.graph_module.graph.nodes
    )
    package = tmp_path / "frame.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.7):
        changed = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in changed),
                "expected": tuple(
                    value.detach().cpu()
                    for value in joint(get_kernel(kind, "torch"), *changed)
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
