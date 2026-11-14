"""Tests for the corresponding EQX source module."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx


def wide_tensor_product_problem(space, shared):
    from equivariantx import co2, o2

    cls = o2 if space == "o2" else co2
    models = {
        backend: cls.TensorProduct(
            "64x1m",
            "64x1m",
            "2x0e+2x0o+2x2m",
            [(0, 0, i, "uvw", True) for i in range(3)],
            backend=backend,
            internal_weights=False,
            shared_weights=shared,
        ).to(device="cuda", dtype=torch.float64)
        for backend in ("torch", "cuda")
    }

    def inputs(rows):
        shapes = (
            (rows, 128),
            (rows, 128),
            (models["torch"].weight_numel,)
            if shared
            else (rows, models["torch"].weight_numel),
        )
        return tuple(
            (
                torch.randn(*shape, dtype=torch.float64, device="cuda") * 0.2
            ).requires_grad_()
            for shape in shapes
        )

    def operation(model, x, y, weight):
        output = model(x, y, weight)
        energy = 1 + output.sin().mean()
        force = torch.autograd.grad(energy, y, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    return models, inputs, operation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("shared", [False, True])
def test_wide_dynamic_tensor_product(space, shared, monkeypatch, tmp_path):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    from equivariantx.ops import config

    torch.manual_seed(921)
    models, inputs, operation = wide_tensor_product_problem(space, shared)
    values = inputs(7)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)

    def joint(x, y, weight):
        return operation(models["cuda"], x, y, weight)

    # Eager initialization is excluded from the captured running graph.
    joint(*values)
    rows = ShapeVar("rows", min=2)
    specs = {"x": TensorSpec((rows, None)), "y": TensorSpec((rows, None))}
    if not shared:
        specs["weight"] = TensorSpec((rows, None))
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec(specs)),
    )(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert "eqx.sparse_bilinear" in graph.code
    assert "eqx.local_channel_product" not in graph.code
    arguments = [node.name for node in graph.graph.nodes if node.op == "placeholder"]
    traced_specs = {name: TensorSpec((rows, None)) for name in arguments[:2]}
    if not shared:
        traced_specs[arguments[2]] = TensorSpec((rows, None))
    compilations = []

    def backend(module, arguments):
        compilations.append(module)
        return module.forward

    run = torch.compile(
        graph,
        backend=backend,
        fullgraph=True,
        dynamic=None,
        dynamic_shapes=ShapesSpec(traced_specs),
    )
    records = []
    for count in (7, 11, 513):
        values = inputs(count)
        for _ in range(3):
            actual = run(*values)
            with monkeypatch.context() as reference:
                reference.setattr(config, "TORCH_REFERENCE", True)
                expected = operation(models["torch"], *values)
            torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
            if shared:
                records.append(
                    {
                        "inputs": tuple(value.detach().cpu() for value in values),
                        "outputs": tuple(value.detach().cpu() for value in expected),
                    }
                )
            values = (
                *values[:2],
                (values[-1] - 0.01 * expected[-1]).detach().requires_grad_(),
            )
    assert len(compilations) == 1
    if not shared:
        return
    rows = torch.export.Dim("rows", min=2)
    initial = tuple(value.cuda() for value in records[0]["inputs"])
    exported = torch.export.export(
        graph, initial, dynamic_shapes=({0: rows}, {0: rows}, None), strict=True
    )
    package, data = tmp_path / "product.pt2", tmp_path / "states.pt"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(value.cuda() for value in row['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), row['outputs'], atol=3e-10, rtol=3e-10)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("nodes", [0, 5])
def test_tensor_product_defaults_to_torch(
    reference, nodes, double_precision, monkeypatch
):
    from equivariantx.ops import config
    from equivariantx.ops.common.channel_product import tensor_product

    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    metadata = repr((1, (2, 2, 1, 2), (((0, 0, 0, 0), 0.7), ((1, 1, 0, 1), -0.2))))
    values = tuple(
        torch.randn(*shape, requires_grad=True) for shape in ((nodes, 2), (1, 2), (1,))
    )
    actual = tensor_product(metadata, *values, 2)
    expected = values[0] * values[1] * values[2] * values[0].new_tensor((0.7, -0.2))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sin().sum(), values),
        torch.autograd.grad(expected.sin().sum(), values),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native CUDA angular products"
)
def test_crossed_angular_product(double_precision):
    from e3nn import o3

    from equivariantx.ops.common.channel_product import crossed_angular_product

    cg = o3.wigner_3j(1, 1, 2, device="cuda")
    entries = tuple((i, j, k, float(cg[i, j, k])) for i, j, k in cg.nonzero().tolist())
    values = (
        torch.randn(3, 5, 1, 3, device="cuda", requires_grad=True),
        torch.randn(1, 1, 7, 3, device="cuda", requires_grad=True),
    )

    def reference(x, y):
        return torch.einsum("...i,...j,ijk->...k", x, y, cg)

    def function(x, y):
        return crossed_angular_product(x, y, entries, 5)

    actual, expected = function(*values), reference(*values)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value.sin().sum(), values, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-10, rtol=2e-10)
        actual, expected = [
            torch.cat([g.flatten() for g in gradients]) / 30 for gradients in (ga, gb)
        ]
    tangents = tuple(torch.randn_like(value) for value in values)
    torch.testing.assert_close(
        torch.func.jvp(function, values, tangents),
        torch.func.jvp(reference, values, tangents),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_native_functional_transforms(double_precision, compiler):
    from equivariantx.ops.common.channel_product import local_product

    metadata = repr((3, (2, 1, 2), (((0, 0, 0), 2.0), ((1, 0, 1), -1.0))))
    values = tuple(
        torch.randn(*shape, device="cuda") for shape in ((5, 2, 3), (1, 1, 3))
    )

    def function(x, w):
        return local_product(metadata, x, w).reshape_as(x)

    def reference(x, w):
        return x * w * x.new_tensor((2.0, -1.0))[None, :, None]

    tangents = tuple(torch.randn_like(value) for value in values)

    def jvp(*values):
        return torch.func.jvp(function, values, tangents)

    torch.compiler.reset()
    run = torch.compile(jvp, backend=compiler, fullgraph=True) if compiler else jvp
    torch.testing.assert_close(
        run(*values), torch.func.jvp(reference, values, tangents)
    )
    batched = tuple(torch.stack((value, value * 0.8)) for value in values)
    torch.testing.assert_close(
        torch.vmap(function)(*batched), torch.vmap(reference)(*batched)
    )
    gradient = torch.func.grad(lambda x, w: function(x, w).sin().sum(), argnums=(0, 1))
    reference_gradient = torch.func.grad(
        lambda x, w: reference(x, w).sin().sum(), argnums=(0, 1)
    )

    def gradient_jvp(*values):
        return torch.func.jvp(gradient, values, tangents)

    run = (
        torch.compile(gradient_jvp, backend=compiler, fullgraph=True)
        if compiler
        else gradient_jvp
    )
    torch.testing.assert_close(
        run(*values), torch.func.jvp(reference_gradient, values, tangents)
    )

    def nested(function, *inputs):
        return torch.func.jvp(
            lambda *primals: torch.func.jvp(function, primals, tangents)[1],
            inputs,
            tangents,
        )

    def nested_jvp(*inputs):
        return nested(function, *inputs)

    run = nested_jvp
    if compiler:
        graph = make_fx(nested_jvp)(*values)
        graph.graph.eliminate_dead_code()
        graph.recompile()
        run = torch.compile(graph, backend=compiler, fullgraph=True)
    torch.testing.assert_close(run(*values), nested(reference, *values))


def test_local_channel_scaling():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.channel_product import local_product

    indices = [0, 1, 2, 1, 2]
    metadata = repr(
        (4, (5, 3, 5), tuple(((i, j, i), 1.0) for i, j in enumerate(indices)))
    )
    x = torch.randn(3, 5, 8, device="cuda", dtype=torch.float64)[
        ..., ::2
    ].requires_grad_()
    w = torch.randn(3, 3, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    actual = local_product(metadata, x, w).view_as(x)
    expected = x * w[:, indices]
    torch.testing.assert_close(actual, expected)
    losses = [z.square().sum() for z in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, (x, w), create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        losses = [sum(g.square().sum() for g in gs) for gs in gradients]


def diagonal_problem(space, shared, dtype, *, channels=3, simple=False):
    from equivariantx import co2, o2

    cls = o2 if space == "o2" else co2
    inputs = f"{channels}x1m+1x1m+{channels}x0e"
    outputs = f"{channels}x0e+{channels}x0o+{channels}x2m+{channels}x1m"
    instructions = [
        (0, 0, 0, "uuu", True, 0.7),
        (0, 1, 2, "u1u", True, 0.6),
        (0, 0, 0, "uuu", True, 1.1),
        (2, 0, 3, "uuu", False, 0.2),
        (0, 1, 1, "u1u", False, 0.3),
    ]
    if simple:
        inputs, outputs = f"{channels}x1m", f"{channels}x0e"
        instructions = [(0, 0, 0, "uuu", True)]
    return cls.TensorProduct(
        inputs,
        inputs,
        outputs,
        instructions,
        internal_weights=False,
        shared_weights=shared,
    ).to(dtype=dtype)


def diagonal_inputs(layer, nodes, dtype, mode="independent"):
    values = [
        (torch.randn(*shape, dtype=dtype) * 0.02)[..., ::2].requires_grad_(
            mode != "frozen_left" or i != 0
        )
        for i, shape in enumerate(
            (
                (nodes, layer.irreps_in1.dim * 2),
                (nodes, layer.irreps_in2.dim * 2),
                (layer.weight_numel * 2,)
                if layer.shared_weights
                else (nodes, layer.weight_numel * 2),
            )
        )
    ]
    if mode == "self":
        values[1] = values[0]
    return tuple(values)


def diagonal_force_loss(function, x, y, weight):
    output = function(x, y, weight)
    energy = output.sin().sum() * 0.01
    coordinates = (x,) if x is y else tuple(t for t in (x, y) if t.requires_grad)
    forces = torch.autograd.grad(energy, coordinates, create_graph=True)
    loss = sum((force + 0.01).square().sum() for force in forces)
    return output, *forces, torch.autograd.grad(loss, weight)[0]


def diagonal_clipping(outputs, arguments):
    saved_inputs = tuple(value.detach().clone() for value in arguments)
    saved_outputs = tuple(value.detach().clone() for value in outputs[:-1])
    outputs[-1].mul_(0.125)
    torch.testing.assert_close(arguments, saved_inputs, atol=0, rtol=0)
    torch.testing.assert_close(outputs[:-1], saved_outputs, atol=0, rtol=0)


def check_diagonal_product(space, shared, dtype, nodes, mode, monkeypatch):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "TORCH_CHANNEL_SLICES", True)
    monkeypatch.setattr(config, "CHANNEL_SLICES_MIN_WIDTH", 3)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    layer = diagonal_problem(space, shared, dtype)
    values = diagonal_inputs(layer, nodes, dtype, mode)
    graph = make_fx(lambda x, y, w: diagonal_force_loss(layer, x, y, w))(*values)
    actual = graph(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = diagonal_force_loss(layer, *values)
    tolerance = 3e-6 if dtype == torch.float32 else 5e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    diagonal_clipping(actual, values)


def check_diagonal_transforms(space, monkeypatch):
    from equivariantx.ops import config
    from equivariantx.tests.utils.testing import check_forward_ad

    monkeypatch.setattr(config, "CHANNEL_SLICES_MIN_WIDTH", 2)
    layer = diagonal_problem(space, True, torch.float64, channels=2, simple=True)
    x, _, weight = diagonal_inputs(layer, 3, torch.float64)

    def high_order(x, weight):
        result = [layer(x, x, weight)]
        loss = result[0].sin().sum() * 0.01
        for _ in range(5):
            gradients = torch.autograd.grad(loss, (x, weight), create_graph=True)
            result.extend(gradients)
            loss = sum(value.sin().sum() * 0.01 for value in gradients)
        return tuple(result)

    graph = make_fx(high_order)(x, weight)
    actual = graph(x, weight)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = high_order(x, weight)
    torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)

    def self_product(x, weight):
        return layer(x, x, weight)

    graph = make_fx(self_product)(x, weight)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    check_forward_ad(graph, self_product, (x, weight))
    batched = torch.stack((x, x * 0.5))
    torch.testing.assert_close(
        torch.vmap(graph, in_dims=(0, None))(batched, weight),
        layer(batched, batched, weight),
        atol=5e-11,
        rtol=5e-11,
    )


def check_diagonal_broadcast(space, shared, monkeypatch):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "CHANNEL_SLICES_MIN_WIDTH", 3)
    layer = diagonal_problem(space, shared, torch.float64)
    x, y, weight = diagonal_inputs(layer, 7, torch.float64)
    x = torch.stack((x[:1], x[:1] * 0.5)).detach().requires_grad_()
    graph = make_fx(lambda x, y, w: diagonal_force_loss(layer, x, y, w))(x, y, weight)
    actual = graph(x, y, weight)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = diagonal_force_loss(layer, x, y, weight)
    torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    diagonal_clipping(actual, (x, y, weight))


def check_diagonal_boundaries(space, monkeypatch):
    from equivariantx import co2, o2
    from equivariantx.ops import config
    from equivariantx.ops.common.torch import channel_product

    cls = o2 if space == "o2" else co2
    monkeypatch.setattr(config, "CHANNEL_SLICES_MIN_WIDTH", 3)
    for output, instructions in (
        ("3x0e+3x2m", [(0, 0, 0, "uuu", False)]),
        ("3x0e", []),
        ("", []),
    ):
        layer = cls.TensorProduct(
            "3x1m",
            "3x1m",
            output,
            instructions,
            internal_weights=False,
            shared_weights=True,
        ).double()
        values = diagonal_inputs(layer, 1, torch.float64)
        graph = make_fx(layer)(*values)
        with monkeypatch.context() as flags:
            flags.setattr(config, "TORCH_REFERENCE", True)
            expected = layer(*values)
        torch.testing.assert_close(graph(*values), expected)
        actual_grad = torch.autograd.grad(
            graph(*values).sum(), values, allow_unused=True
        )
        expected_grad = torch.autograd.grad(expected.sum(), values, allow_unused=True)
        torch.testing.assert_close(actual_grad, expected_grad)
    layer = diagonal_problem(space, True, torch.float64)
    values = diagonal_inputs(layer, 3, torch.float64)

    def reject(*args):
        raise AssertionError("CPU slice schedule should not be selected")

    monkeypatch.setattr(channel_product, "slice_product", reject)
    layer(*values)
    monkeypatch.setattr(config, "TORCH_CHANNEL_SLICES", False)
    make_fx(layer)(*values)
    monkeypatch.setattr(config, "TORCH_CHANNEL_SLICES", True)
    monkeypatch.setattr(config, "POLYNOMIAL_INLINE", False)
    make_fx(layer)(*values)
    monkeypatch.setattr(config, "POLYNOMIAL_INLINE", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    make_fx(layer)(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    layer.kernel_metadata = None
    make_fx(layer)(*values)
    del layer.kernel_metadata
    make_fx(layer)(*values)


@pytest.mark.parametrize("nodes", [0, 1, 7])
@pytest.mark.parametrize(
    "mask", [(True, True, True, False), (False, False, True, False)]
)
def test_cpu_slice_required_masks(nodes, mask):
    from equivariantx.ops.common.polynomial import polynomial_adjoint
    from equivariantx.ops.common.torch.channel_product import reference, slice_product

    metadata = repr(
        (
            3,
            (2, 2, 1, 2),
            (
                ((0, 0, 0, 0), 1.0),
                ((1, 1, 0, 0), 2.0),
                ((0, -1, -1, 1), 0.5),
                ((1, 0, -1, 1), -0.3),
            ),
        )
    )
    program = repr((((0, 1, 2, 3), 3, 0, 0),))
    operands = [
        torch.randn(n, width, dtype=torch.float64, requires_grad=True)
        for n, width in ((nodes, 6), (1, 6), (1, 3))
    ]
    operands.append(operands[0].new_empty((0, 6)))
    cotangents = [torch.randn(nodes, 6, dtype=torch.float64, requires_grad=True)]
    for _ in range(3):
        program, operands, _ = polynomial_adjoint(program, operands, cotangents, mask)
        actual = slice_product(metadata, program, operands)
        torch.testing.assert_close(actual, reference(metadata, program, operands))
        cotangents = [torch.randn_like(value, requires_grad=True) for value in actual]
        mask = tuple(value.requires_grad for value in operands)


def check_diagonal_dynamic(space, shared, monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    from equivariantx.ops import config

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    layer = diagonal_problem(space, shared, torch.float64, channels=64, simple=True)
    values = tuple(
        value.detach().contiguous().requires_grad_()
        for value in diagonal_inputs(layer, 7, torch.float64)
    )

    def joint(x, y, w):
        return diagonal_force_loss(layer, x, y, w)

    nodes = ShapeVar("nodes", min=0, optimization_hint=7)
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec([nodes, None]),
                    "y": TensorSpec([nodes, None]),
                    "w": TensorSpec([None] if shared else [nodes, None]),
                }
            )
        ),
    )(*values)
    compilations = []

    def backend(module, inputs):
        compilations.append(module)
        return module.forward

    run = torch.compile(graph, backend=backend, dynamic=True, fullgraph=True)
    for count in (7, 11, 9, 0, 1):
        if count == 0:
            assert len(compilations) == 1
        values = tuple(
            value.detach().contiguous().requires_grad_()
            for value in diagonal_inputs(layer, count, torch.float64)
        )
        for _ in range(2):
            with monkeypatch.context() as flags:
                flags.setattr(config, "TORCH_REFERENCE", True)
                expected = joint(*values)
            arguments = tuple(value.detach() for value in values)
            actual = run(*arguments)
            torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
            diagonal_clipping(actual, arguments)
            values = (
                *values[:2],
                (values[-1] - 0.001 * expected[-1]).detach().requires_grad_(),
            )
