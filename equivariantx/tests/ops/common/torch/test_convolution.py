"""Frame convolution roles, functional transforms and export."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import conv, o2
from equivariantx.ops import get_kernel
from equivariantx.ops.decompositions import torch_decompositions
from equivariantx.ops.common.o2_o3 import contraction
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.parametrize("kind", ["o2", "o3"])
@pytest.mark.parametrize("width", [64, 128, 1024])
@pytest.mark.parametrize("shared", [False, True])
def test_projected_schedule(kind, width, shared, double_precision, monkeypatch):
    from equivariantx.ops import config
    from equivariantx.tests.ops.common.torch.test_o3 import (
        convolution_case as spherical_case,
    )

    if kind == "o2":
        metadata, values, source, target = convolution_case(shared=shared, device="cpu")
        kernel, output = "contraction", 6
    else:
        _, metadata, values, source, target = spherical_case(
            True, shared=shared, device="cpu"
        )
        kernel, output = "o3_contraction", 4
    values = (
        values[0],
        torch.randn(values[1].shape[0], width) * 0.2,
        torch.randn(width, values[2].shape[1]) / width**0.5,
        *values[3:],
    )
    values = tuple(value.requires_grad_() for value in values)
    program = repr(((tuple(range(len(values))), False, ((output, 0), (1, 1), (2, 2))),))
    actual = get_kernel(kernel, "torch")(metadata, program, source, target, values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = get_kernel(kernel, "torch")(metadata, program, source, target, values)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    actual, expected = [
        sum(value.sin().sum() for value in outputs) for outputs in (actual, expected)
    ]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value, values, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [sum(g.sin().sum() for g in gs) / width for gs in gradients]


def convolution_case(edges=7, shared=False, projected=True, device="cuda"):
    tp = o2.O3TensorProduct(
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o+2x1o",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 2, "uvu", False),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    module = conv.O2O3TensorProductConv(tp, method="wigner").to(device)
    rows = 1 if shared else edges
    matrix_width = sum((2 * degree + 1) ** 2 for degree in range(tp.lmax + 1))
    shapes = (
        (3, tp.input_dim),
        (rows, 4 if projected else tp.weight_numel),
        (4 if projected else 0, tp.weight_numel),
        (rows, matrix_width),
        (rows, matrix_width),
        (rows, tp.num_harmonics),
        (5, module.output_dim),
    )
    values = tuple(torch.randn(shape, device=device) * 0.2 for shape in shapes)
    source, target = (
        torch.arange(edges, device=device) % 3,
        torch.arange(edges, device=device) % 5,
    )
    return module.kernel_metadata, values, source, target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA convolution comparison")
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("edges", [0, 7])
def test_transposed_roles(shared, projected, edges, double_precision):
    metadata, values, source, target = convolution_case(edges, shared, projected)
    values = tuple(value.requires_grad_() for value in values)
    for role in range(7):
        program = repr(((tuple(range(7)), False, ((role, 0),)),))
        actual = contraction(metadata, program, source, target, list(values))[0]
        expected = get_kernel("contraction", "torch")(
            metadata, program, source, target, values
        )[0]
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        active = tuple(value for i, value in enumerate(values) if i != role)
        for _ in range(3):
            gradients = [
                torch.autograd.grad(
                    value.sin().sum(), active, create_graph=True, allow_unused=True
                )
                for value in (actual, expected)
            ]
            gradients = [
                tuple(
                    torch.zeros_like(x) if g is None else g for x, g in zip(active, gs)
                )
                for gs in gradients
            ]
            torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
            actual, expected = [
                sum(g.sum() for g in gs) / max(sum(g.numel() for g in gs), 1)
                for gs in gradients
            ]
            if not actual.requires_grad:
                break


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA convolution transforms")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_transforms(compiler, double_precision):
    metadata, values, source, target = convolution_case()
    program = repr((((0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),))
    values = values[0], values[1], values[2], values[3], values[5], values[6]

    def function(*inputs):
        return contraction(
            metadata, program, inputs[-1][0], inputs[-1][1], list(inputs[:-1])
        )[0]

    def reference(*inputs):
        return get_kernel("contraction", "torch")(
            metadata, program, inputs[-1][0], inputs[-1][1], inputs[:-1]
        )[0]

    assert_functional_derivatives(
        function, reference, (*values, torch.stack((source, target))), compiler
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA convolution export")
def test_force_derivatives_aoti(double_precision, tmp_path):
    metadata, values, source, target = convolution_case()
    program = repr((((0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),))
    placeholder = values[6]
    inputs = tuple(
        value.requires_grad_()
        for value in (values[0], values[1], values[2], values[3], values[5])
    )

    def joint(function, *inputs):
        output = function(metadata, program, source, target, [*inputs, placeholder])[0]
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, inputs[3], create_graph=True)[0]
        gradients = torch.autograd.grad(
            energy + (force - 0.01).square().sum(), inputs[:3]
        )
        return energy, force, *gradients

    graph = make_fx(
        lambda *inputs: joint(contraction, *inputs),
        decomposition_table=torch_decompositions(),
    )(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, inputs, strict=True)
    assert all(
        not str(node.target).startswith("eqx.")
        for node in exported.graph_module.graph.nodes
    )
    package = tmp_path / "convolution.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.7):
        changed = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in changed),
                "expected": tuple(
                    value.detach().cpu()
                    for value in joint(get_kernel("contraction", "torch"), *changed)
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
