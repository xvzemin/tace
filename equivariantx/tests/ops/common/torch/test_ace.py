"""Ordinary sparse ACE schedules and every transposed factor role."""

import os
import subprocess
import sys

import pytest
import torch

from equivariantx.ops import config, get_kernel
from equivariantx.ops.common.polynomial import polynomial_adjoint
from equivariantx.utils.metadata import encode_plans


def coefficient_case(kind, nodes, channels, elements, device):
    values = [
        (torch.randn(*shape, device=device) * 0.2).requires_grad_()
        for shape in (
            (nodes, 4 * channels),
            (nodes, 4 * channels),
            (elements, channels**2),
            (nodes, 2 * channels),
        )
    ]
    if kind == "bilinear":
        values.append(torch.randn(nodes, 2 * channels, device=device).requires_grad_())
    records = tuple(
        (
            (
                2 * k * channels,
                2 * k * channels + 1,
                0,
                1,
                channels,
                channels,
                2,
                2,
                2,
                k * channels,
            ),
            coefficient,
        )
        for k, coefficient in enumerate((0.7, -0.3))
    )
    schedules = tuple((((1, channels, 2), records),) for _ in values)
    metadata = repr((tuple(value.shape[1] for value in values), True, schedules))
    mapping = tuple(range(len(values)))
    if kind == "bilinear":
        metadata = encode_plans([metadata])
        program = repr(((0, mapping, 3, 0),))
    else:
        program = repr(((mapping, 3, 0),))
    types = torch.arange(nodes, device=device) % max(elements, 2)
    return metadata, program, values, types


@pytest.mark.parametrize("kind", ["ace", "bilinear"])
@pytest.mark.parametrize(
    "device,nodes,channels,elements",
    [
        ("cpu", 7, 3, 2),
        ("cpu", 7, 3, 1),
        ("cuda", 0, 64, 89),
        ("cuda", 2048, 64, 89),
        ("cuda", 2048, 64, 1),
    ],
)
def test_accumulated_coefficients(
    kind, device, nodes, channels, elements, double_precision, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA grouping")
    metadata, program, values, types = coefficient_case(
        kind, nodes, channels, elements, device
    )
    name = "ace_contract" if kind == "ace" else "bilinear_ace"
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = get_kernel(name, "torch")(metadata, program, types, values)[0]
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = get_kernel(name, "torch")(metadata, program, types, values)[0]
    active = (*values[:3], *values[4:])
    directions = tuple(torch.randn_like(value) for value in active)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
        gradients = [
            torch.autograd.grad(
                output.sin().sum() / max(output.numel(), 1), active, create_graph=True
            )
            for output in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-11, rtol=1e-9)
        actual, expected = [
            sum((g * direction).sum() for g, direction in zip(gs, directions)) / 20
            for gs in gradients
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouping")
@pytest.mark.parametrize("kind", ["ace", "bilinear"])
@pytest.mark.parametrize("elements", [1, 89])
def test_grouped_transposed_roles(kind, elements, double_precision):
    from equivariantx.ops.common.torch.ace import contract

    metadata, _, values, types = coefficient_case(kind, 2048, 64, elements, "cuda")
    mapping = tuple(range(len(values)))
    for role in range(len(values)):
        program = repr(
            ((0, mapping, role, 0),) if kind == "bilinear" else ((mapping, role, 0),)
        )
        actual = contract(kind, metadata, program, types, values, optimized=True)[0]
        expected = contract(kind, metadata, program, types, values, optimized=False)[0]
        for _ in range(2):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
            gradients = [
                torch.autograd.grad(
                    output.sin().sum() / max(output.numel(), 1),
                    values,
                    create_graph=True,
                )
                for output in (actual, expected)
            ]
            torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-9)
            actual, expected = [sum(g.sum() for g in gs) / 20 for gs in gradients]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouping")
@pytest.mark.parametrize("compiler", [None, "inductor"])
def test_grouped_functional_transforms(compiler, double_precision):
    from equivariantx.ops.common.torch.ace import bilinear_contract, bilinear_reference
    from equivariantx.tests.ops.testing import assert_functional_derivatives

    metadata, program, values, types = coefficient_case("bilinear", 2048, 64, 2, "cuda")

    def function(*inputs):
        return bilinear_contract(metadata, program, inputs[-1], inputs[:-1])[0]

    def reference(*inputs):
        return bilinear_reference(metadata, program, inputs[-1], inputs[:-1])[0]

    assert_functional_derivatives(function, reference, (*values, types), compiler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
def test_grouped_force_aoti(tmp_path, double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops.common.torch.ace import bilinear_contract, bilinear_reference

    metadata, program, values, types = coefficient_case("bilinear", 2048, 64, 2, "cuda")

    def joint(function, *inputs):
        values, types = inputs[:-1], inputs[-1]
        output = function(metadata, program, types, values)[0]
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, values[0], create_graph=True)[0]
        loss = energy + (force - 0.01).square().sum()
        return (
            loss,
            force,
            *torch.autograd.grad(loss, (values[1], values[2], values[4])),
        )

    graph = make_fx(lambda *inputs: joint(bilinear_contract, *inputs))(*values, types)
    inputs = (*[value.detach() for value in values], types)
    exported = torch.export.export(graph, inputs)
    package = tmp_path / "grouped.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale, indices in ((1.0, types), (0.7, torch.zeros_like(types))):
        changed = tuple((value.detach() * scale).requires_grad_() for value in values)
        cases.append(
            {
                "inputs": (*[value.detach().cpu() for value in changed], indices.cpu()),
                "expected": tuple(
                    value.detach().cpu()
                    for value in joint(bilinear_reference, *changed, indices)
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
    torch.testing.assert_close(tuple(value.cpu() for value in actual), case['expected'], atol=1e-9, rtol=1e-9)
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


@pytest.mark.parametrize("kind", ["unweighted", "weighted", "gated"])
@pytest.mark.parametrize("nodes", [0, 5])
@pytest.mark.parametrize("shared", [False, True])
def test_products_and_transposes(kind, nodes, shared, double_precision):
    x, y = [torch.randn(nodes, 3, requires_grad=True) for _ in range(2)]
    types = torch.arange(nodes) % 2
    weighted = kind != "unweighted"
    output_dim = 2 if weighted else 3
    operands = [x, y]
    record = (0, 0, 0, 0, 3, 2 if weighted else 0, 1, 1, 1, 0)
    dims = [3, 3]
    expected = x * y
    continuous = [x, y]
    if weighted:
        weight = torch.randn(1 if shared else 2, 6, requires_grad=True)
        operands.append(weight)
        continuous.append(weight)
        dims.append(6)
    operands.append(x.new_empty((0, output_dim)))
    dims.append(output_dim)
    role = len(operands) - 1
    if kind == "gated":
        gates = torch.randn(nodes, 3, requires_grad=True)
        operands.append(gates)
        continuous.append(gates)
        dims.append(3)
        expected = expected * gates
    if weighted:
        indices = torch.zeros_like(types) if shared else types
        matrix = weight[indices].reshape(nodes, 3, 2)
        expected = (expected.unsqueeze(1) @ matrix).squeeze(1)
    expected = expected * 0.7
    schedules = tuple((((0, 1, 1), ((record, 0.7),)),) for _ in operands)
    metadata = repr((tuple(dims), weighted, schedules))
    mapping = tuple(range(len(operands)))
    program = repr(((mapping, role, 0),))
    backend_kind = "ace"
    name = "ace_contract"
    if kind == "gated":
        metadata = encode_plans([metadata])
        program = repr(((0, mapping, role, 0),))
        backend_kind, name = "bilinear", "bilinear_ace"
    evaluate = get_kernel(name, "torch")
    actual = evaluate(metadata, program, types, operands)[0]
    torch.testing.assert_close(actual, expected)
    seed = torch.randn_like(expected)
    adjoint, values, destinations = polynomial_adjoint(
        program, operands, (seed,), [x.requires_grad for x in operands], backend_kind
    )
    result = evaluate(metadata, adjoint, types, values)
    gradients = torch.autograd.grad(expected, continuous, seed, create_graph=True)
    for value, gradient in zip(continuous, gradients):
        index = next(i for i, operand in enumerate(operands) if operand is value)
        torch.testing.assert_close(result[destinations[index]], gradient)
    for _ in range(4):
        actual, expected = [
            torch.cat(
                [
                    gradient.flatten()
                    for gradient in torch.autograd.grad(
                        value.sin().sum(), continuous, create_graph=True
                    )
                ]
            )
            / 10
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
