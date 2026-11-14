"""O3 factor transposes, solid-harmonic derivatives and compiler visibility."""

import os
import subprocess
import sys

import pytest
import torch
from e3nn import o3 as e3nn_o3
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3, conv, o3
from equivariantx.ops import get_kernel
from equivariantx.ops.decompositions import torch_decompositions
from equivariantx.ops.common.torch.o3 import (
    angular_values,
    coupling_data,
    harmonic_coefficients,
    harmonic_values,
    transverse_values,
)
from equivariantx.ops.common.o3 import contraction
from equivariantx.tests.ops.testing import assert_functional_derivatives


def convolution_case(geometric, edges=7, shared=False, projected=True, device="cuda"):
    product = o3.TensorProduct(
        "2x0e+2x1o",
        "0e+2x1o+2x2e",
        "2x0e+2x1o+2x1o+2x2e",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 2, "uvu", False),
            (0, 2, 3, "uvu", True),
            (1, 2, 1, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    module = conv.O3TensorProductConv(product, normalize=False).to(device)
    rows = 1 if shared else edges
    shapes = [
        (3, module.input_dim),
        (rows, 4 if projected else module.weight_numel),
        (4 if projected else 0, module.weight_numel),
        (rows, module.amplitude_dim if geometric else module.edge_dim),
        (5, module.output_dim),
    ]
    if geometric:
        shapes.append((rows, 3))
    values = tuple(
        (torch.randn(shape, device=device) * 0.2).requires_grad_() for shape in shapes
    )
    source, target = (
        torch.arange(edges, device=device) % 3,
        torch.arange(edges, device=device) % 5,
    )
    metadata = module.harmonic_metadata if geometric else module.kernel_metadata
    return module, metadata, values, source, target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA O3 comparison")
@pytest.mark.parametrize("geometric", [False, True])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("edges", [0, 7])
def test_transposed_roles(geometric, projected, shared, edges, double_precision):
    _, metadata, values, source, target = convolution_case(
        geometric, edges, shared, projected
    )
    if geometric:
        values = (*values, torch.randn_like(values[5], requires_grad=True) * 0.2)
    for role in (*range(5), *((6,) if geometric else ())):
        program = repr(((tuple(range(len(values))), False, ((role, 0),)),))
        actual = contraction(metadata, program, source, target, list(values))[0]
        expected = get_kernel("o3_contraction", "torch")(
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


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_harmonic_directional_derivatives(degree, normalization, double_precision):
    vector = torch.randn(4, 3) * 0.3
    directions = tuple(torch.randn_like(vector) * 0.3 for _ in range(4))
    function = e3nn_o3.SphericalHarmonics(
        degree, normalize=False, normalization=normalization
    )
    coefficients = harmonic_coefficients(2 * degree + 1, normalization)
    for rank in range(5):
        torch.testing.assert_close(
            harmonic_values(coefficients, vector, directions[:rank]),
            function(vector),
            atol=1e-10,
            rtol=1e-10,
        )
        previous, direction = function, directions[min(rank, 3)]

        def function(x, previous=previous, direction=direction):
            return torch.func.jvp(previous, (x,), (direction,))[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA O3 transforms")
@pytest.mark.parametrize("geometric", [False, True])
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_transforms(geometric, compiler, double_precision):
    module, metadata, values, source, target = convolution_case(geometric)
    program = repr(((tuple(range(len(values))), False, ((4, 0),)),))

    def function(*inputs):
        return contraction(
            metadata, program, inputs[-1][0], inputs[-1][1], list(inputs[:-1])
        )[0]

    def reference(*inputs):
        x, radial, projection, attributes, _, *geometry = inputs[:-1]
        if geometric:
            sections, position = [], 0
            for multiplicity, degree in module.harmonic_entries:
                harmonic = e3nn_o3.spherical_harmonics(
                    degree,
                    geometry[0],
                    normalize=False,
                    normalization=module.normalization,
                )
                sections.append(
                    (
                        harmonic[:, :, None]
                        * attributes[:, None, position : position + multiplicity]
                    ).flatten(1)
                )
                position += multiplicity
            attributes = torch.cat(sections, dim=1)
        return module.reference(x, attributes, radial, projection, inputs[-1], 5)

    assert_functional_derivatives(
        function, reference, (*values, torch.stack((source, target))), compiler
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("symmetric", [False, True])
def test_cartesian_harmonics(degree, symmetric, double_precision):
    from equivariantx.co3.symmetric import symmetric_powers
    from equivariantx.conv.co3.polynomials import component_powers

    kind = "cartesian_symmetric" if symmetric else "cartesian"
    width = (degree + 1) * (degree + 2) // 2 if symmetric else 3**degree
    vector = torch.randn(5, 3, requires_grad=True) * 0.3
    coefficients = harmonic_coefficients(width, (kind, "component"))
    actual = harmonic_values(coefficients, vector, ())
    expected = (
        co3.CartesianHarmonics(degree, normalize=False, normalization="component")(
            vector
        )
        + vector[:, :1] * 0
    )
    if symmetric:
        indices = [component_powers(i, degree) for i in range(3**degree)]
        expected = expected[
            :, [indices.index(power) for power in symmetric_powers(degree)]
        ]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sin().sum(), vector),
        torch.autograd.grad(expected.sin().sum(), vector),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA edge-local O3")
@pytest.mark.parametrize("geometric", [False, True])
def test_edge_local_and_direction_offset(geometric, double_precision):
    from equivariantx.utils.metadata import parse_metadata

    _, metadata, values, source, target = convolution_case(geometric)
    values = [values[0][source], *values[1:4], values[4][target], *values[5:]]
    if geometric:
        values[5] = torch.nn.functional.pad(values[5], (2, 1))
        metadata = repr((*parse_metadata(metadata), False, False, 2))
    program = repr(((tuple(range(len(values))), False, ((4, 0),)),))
    actual = contraction(metadata, program, None, None, values)[0]
    expected = get_kernel("o3_contraction", "torch")(
        metadata, program, None, None, values
    )[0]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    active = (*values[:4], *values[5:])
    for _ in range(3):
        gradients = [
            torch.autograd.grad(value.sin().sum(), active, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-10)
        actual, expected = [
            sum(value.square().sum() for value in gradient) for gradient in gradients
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA O3 export")
@pytest.mark.parametrize(
    "geometric", [False, True, "baseline", "generator", "recurrence", "angular"]
)
def test_force_derivatives_aoti(geometric, double_precision, tmp_path):
    from equivariantx.ops.common.torch.o3 import reference
    from equivariantx.utils.metadata import parse_metadata

    _, metadata, values, source, target = convolution_case(geometric)
    if isinstance(geometric, str):
        description = parse_metadata(metadata)
        metadata = (
            repr((*description, False, True))
            if geometric == "angular"
            else repr((*description[:3], ("transverse", "component", geometric)))
        )
    placeholder = values[4].detach()
    inputs = (*values[:4], *values[5:])
    program = repr(((tuple(range(len(values))), False, ((4, 0),)),))

    def joint(function, *inputs):
        operands = [*inputs[:4], placeholder, *inputs[4:]]
        output = function(metadata, program, source, target, operands)[0]
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, inputs[-1], create_graph=True)[0]
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
    package = tmp_path / "o3.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.7):
        changed = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in changed),
                "expected": tuple(
                    value.detach().cpu() for value in joint(reference, *changed)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA O3 tracing")
@pytest.mark.parametrize("method", [None, "baseline"])
def test_baseline_transverse_lowering(method, double_precision):
    from equivariantx.utils.metadata import parse_metadata

    _, metadata, values, source, target = convolution_case(True)
    description = parse_metadata(metadata)
    geometry = ("transverse", "component") + (() if method is None else (method,))
    metadata = repr((*description[:3], geometry))
    program = repr(((tuple(range(len(values))), False, ((4, 0),)),))

    def function(*values):
        return contraction(metadata, program, source, target, list(values))[0]

    graph = make_fx(
        function,
        decomposition_table=torch_decompositions(),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*values)
    assert not any(
        node.target == torch.ops.eqx.o3_contraction.default
        for node in graph.graph.nodes
    )
    torch.testing.assert_close(
        graph(*values), function(*values), atol=1e-11, rtol=1e-11
    )


@pytest.mark.parametrize("degrees", [(0, 2, 2), (2, 1, 3), (3, 1, 2), (2, 4, 2)])
@pytest.mark.parametrize("method", ["generator", "recurrence"])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_transverse_polynomial(degrees, method, normalization, double_precision):
    from equivariantx.co2.spherical import SphericalCoupling

    l1, l2, l3 = degrees
    features = torch.randn(4, 2 * l1 + 1, requires_grad=True)
    vector = torch.randn(4, 3, requires_grad=True)
    unit = vector / vector.norm(dim=-1, keepdim=True)
    actual = torch.einsum(
        "ea,eac->ec",
        features,
        transverse_values(l1, l2, l3, normalization, method, unit, ()),
    )
    reference = SphericalCoupling(
        l1,
        l2,
        l3,
        normalization,
        "recurrence" if method == "recurrence" else "chebyshev",
    )
    expected = reference(features, vector)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        gradients = [
            torch.autograd.grad(
                value.sin().sum(), (features, vector), create_graph=True
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [sum(g.sum() for g in gs) / 20 for gs in gradients]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transverse comparison")
@pytest.mark.parametrize("degrees", [(1, 2, 1), (2, 3, 3), (3, 4, 3)])
@pytest.mark.parametrize("method", ["baseline", "generator", "recurrence"])
def test_transverse_roles(degrees, method, double_precision):
    from equivariantx.utils.metadata import parse_metadata

    l1, l2, l3 = degrees
    product = o3.TensorProduct(
        f"2x{l1}e",
        f"2x{l2}e",
        f"2x{l3}e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    module = conv.O3TensorProductConv(product, normalize=False).cuda()
    description = parse_metadata(module.harmonic_metadata)
    metadata = repr((*description[:3], ("transverse", "component", method)))
    source, target = (
        torch.arange(7, device="cuda") % 3,
        torch.arange(7, device="cuda") % 5,
    )
    shapes = [
        (3, module.input_dim),
        (7, 4),
        (4, module.weight_numel),
        (7, module.amplitude_dim),
        (5, module.output_dim),
        (7, 3),
        (7, 3),
        (7, 3),
    ]
    operands = tuple(
        (torch.randn(shape, device="cuda") * 0.3).requires_grad_() for shape in shapes
    )
    for rank in range(3):
        values = operands[: 6 + rank]
        for role in (*range(5), *range(6, 6 + rank)):
            program = repr(((tuple(range(len(values))), False, ((role, 0),)),))
            actual = contraction(metadata, program, source, target, list(values))[0]
            expected = get_kernel("o3_contraction", "torch")(
                metadata, program, source, target, values
            )[0]
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            if role == 4 and rank == 0:
                active = (*values[:4], *values[5:])
                for _ in range(3):
                    gradients = [
                        torch.autograd.grad(
                            value.sin().sum(), active, create_graph=True
                        )
                        for value in (actual, expected)
                    ]
                    torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
                    actual, expected = [
                        sum(g.sum() for g in gs) / 20 for gs in gradients
                    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transverse lowering")
@pytest.mark.parametrize("method", ["baseline", "generator", "recurrence"])
@pytest.mark.parametrize("compiler", [None, "inductor"])
def test_transverse_lowering(method, compiler, double_precision):
    from equivariantx.utils.metadata import parse_metadata

    _, metadata, values, source, target = convolution_case(True)
    description = parse_metadata(metadata)
    metadata = repr((*description[:3], ("transverse", "component", method)))
    program = repr(((tuple(range(len(values))), False, ((4, 0),)),))

    def function(*values):
        return contraction(metadata, program, source, target, list(values))[0]

    coupling_data.cache_clear()
    graph = make_fx(
        function,
        decomposition_table=torch_decompositions(),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*values)
    assert not any(
        node.target == torch.ops.eqx.o3_contraction.default
        for node in graph.graph.nodes
    )
    torch.testing.assert_close(
        graph(*values), function(*values), atol=1e-11, rtol=1e-11
    )

    def wrapped(*inputs):
        return contraction(
            metadata, program, inputs[-1][0], inputs[-1][1], list(inputs[:-1])
        )[0]

    def reference(*inputs):
        return get_kernel("o3_contraction", "torch")(
            metadata, program, inputs[-1][0], inputs[-1][1], inputs[:-1]
        )[0]

    assert_functional_derivatives(
        wrapped, reference, (*values, torch.stack((source, target))), compiler
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_ordered_lie_derivatives(degree, double_precision):
    vectors = torch.randn(4, 3) * 0.3
    directions = tuple(torch.randn_like(vectors) for _ in range(3))
    function = e3nn_o3.SphericalHarmonics(
        degree, normalize=False, normalization="component"
    )
    for rank in range(4):
        torch.testing.assert_close(
            angular_values(2 * degree + 1, "component", vectors, directions[:rank]),
            function(vectors),
            atol=1e-10,
            rtol=1e-10,
        )
        previous, direction = function, directions[min(rank, 2)]

        def function(x, previous=previous, direction=direction):
            return torch.func.jvp(previous, (x,), (torch.linalg.cross(direction, x),))[
                1
            ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Lie derivatives")
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_angular_roles_and_cartesian_gradients(rank, double_precision):
    from equivariantx.utils.metadata import parse_metadata

    _, metadata, values, source, target = convolution_case(True)
    metadata = repr((*parse_metadata(metadata), False, True))
    values = (
        *values,
        *(torch.randn_like(values[5], requires_grad=True) * 0.3 for _ in range(rank)),
    )
    for role in (*range(5), *range(6, 6 + rank)):
        program = repr(((tuple(range(len(values))), False, ((role, 0),)),))

        def function(*values):
            return contraction(metadata, program, source, target, list(values))[0]

        actual = function(*values)
        expected = get_kernel("o3_contraction", "torch")(
            metadata, program, source, target, values
        )[0]
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        if role != 4:
            continue
        graph = make_fx(function, decomposition_table=torch_decompositions())(*values)
        assert not any(
            node.target == torch.ops.eqx.o3_contraction.default
            for node in graph.graph.nodes
        )
        torch.testing.assert_close(graph(*values), expected, atol=1e-11, rtol=1e-11)
        active = (*values[:4], *values[5:])
        for _ in range(3):
            gradients = [
                torch.autograd.grad(value.sin().sum(), active, create_graph=True)
                for value in (actual, expected)
            ]
            torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
            actual, expected = [sum(g.sum() for g in gs) / 20 for gs in gradients]
