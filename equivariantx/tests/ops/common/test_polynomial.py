"""Standalone compiled JVPs of native polynomial gradient programs."""

import os
import subprocess
import sys

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.mbe.ace import (
    BilinearContraction,
    UuuFullyConnectedSymmetricContraction,
    UuuSymmetricContraction,
)
from equivariantx.o3 import MoEElementLinear
from equivariantx.tests.utils.testing import package_script


def test_inline_work_counts_vectorized_batches():
    from equivariantx.ops.common.compilation import polynomial_work

    rows = tuple(((0, 1), i, 0, 1.0) for i in range(400))
    metadata = repr((2, 1, 400, 2, rows))
    program = repr((((0, 0, 1, 2), 3, 0, 0),))
    assert polynomial_work("symmetric", metadata, program) == 4
    metadata = repr((1, (400, 400, 400), tuple(((i, i, i), 1.0) for i in range(400))))
    program = repr((((0, 1, 2), 2, 0, 0),))
    assert polynomial_work("channel", metadata, program) == 3


def test_inline_budget(monkeypatch):
    from equivariantx.ops import config
    from equivariantx.ops.common.compilation import should_inline

    def unexpected(*args):
        pytest.fail("Eager or disabled programs must not estimate graph work")

    assert not should_inline(True, unexpected)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not should_inline(False, unexpected)
    monkeypatch.setattr(config, "COMPILE_MAX_INLINE_WORK", 4)
    assert should_inline(True, int, 4)
    assert not should_inline(True, int, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize(
    "kind,native", [("symmetric", False), ("linear", False), ("linear", True)]
)
def test_gradient_jvp_aoti(kind, native, tmp_path, double_precision, monkeypatch):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "LINEAR_NATIVE", native)
    if kind == "symmetric":
        modules = [
            UuuSymmetricContraction.from_entries(
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
    )(*inputs)
    # Forward AD emits unused boolean shape checks; export guards input shapes.
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, inputs, strict=True)
    assert all(
        not str(node.target).startswith("eqx.")
        or (native and str(node.target) == "eqx.element_linear.default")
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
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize("inline_limit", [0, 256])
@pytest.mark.parametrize(
    "kind,native",
    [
        ("symmetric", False),
        ("linear", False),
        ("linear", True),
        ("bilinear", False),
        ("ace", False),
    ],
)
def test_default_force_gradient_aoti_updates(
    kind, native, inline_limit, tmp_path, double_precision, monkeypatch
):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "LINEAR_NATIVE", native)
    monkeypatch.setattr(config, "COMPILE_MAX_INLINE_WORK", inline_limit)
    types = torch.tensor([0, 1, 0], device="cuda")
    if kind == "symmetric":
        module = UuuSymmetricContraction.from_entries(
            "2x0e",
            "2x0e",
            [((0,), 0, 0, 0.4), ((0, 0, 0), 1, 0, -0.1)],
            2,
            backend="cuda",
        ).cuda()
        shapes = ((3, 2, 1), (2, 2, 2))

        def apply(x, weight):
            return module(x, weight, types)

    elif kind == "linear":
        linear = o3.Linear(
            "2x0e+2x1o", "3x0e+3x1o", internal_weights=False, shared_weights=False
        )
        module = MoEElementLinear(linear, 2, backend="cuda").cuda()
        shapes = ((3, 16), (2, 2, linear.weight_numel))

        def apply(x, weight):
            return module(x, weight, types)

    else:
        product = o3.TensorProduct(
            "2x0e",
            "2x0e",
            "2x0e",
            [(0, 0, 0, "uuu", kind == "bilinear")],
            internal_weights=False,
            shared_weights=False,
        )
        linear = o3.Linear("2x0e", "3x0e", internal_weights=False, shared_weights=False)
        if kind == "bilinear":
            module = BilinearContraction(product, linear, backend="cuda").cuda()
            shapes = (
                (3, 2),
                (3, 2),
                (3, product.weight_numel),
                (2, linear.weight_numel),
            )

            def apply(x, base, gates, weight):
                return module(x, base, gates, weight, types)

        else:
            module = UuuFullyConnectedSymmetricContraction(
                [product], [linear, linear], backend="cuda"
            ).cuda()
            shapes = ((3, 2), (2, linear.weight_numel), (2, linear.weight_numel))

            def apply(x, *weights):
                return module(x, weights, types)

    values = tuple(
        (torch.randn(shape, device="cuda") * 0.1 + 0.2).requires_grad_()
        for shape in shapes
    )

    def joint(*inputs):
        energy = apply(*inputs).sin().sum() + 1
        force = -torch.autograd.grad(energy, inputs[0], create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, inputs[-1])[0]

    graph = make_fx(joint)(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    operators = {
        str(node.target)
        for node in graph.graph.nodes
        if str(node.target).startswith("eqx.")
    }
    operator = {
        "symmetric": "symmetric_contract",
        "linear": "element_linear",
        "bilinear": "bilinear_ace",
        "ace": "ace_contract",
    }[kind]
    assert operators == (
        {f"eqx.{operator}.default"} if native or not inline_limit else set()
    )
    exported = torch.export.export(graph, tuple(value.detach() for value in values))
    package = tmp_path / "polynomial_force.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    module.backend = "torch"
    rows = []
    for _ in range(3):
        expected = joint(*values)
        rows.append(
            dict(
                inputs=tuple(value.detach().cpu() for value in values),
                expected=tuple(value.detach().cpu() for value in expected),
            )
        )
        weight = (values[-1] - 0.001 * expected[-1]).detach().requires_grad_()
        values = (*values[:-1], weight)
    data = tmp_path / "cases.pt"
    torch.save(rows, data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            package_script("""
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    actual=run(*(value.cuda() for value in row['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),row['expected'],atol=2e-9,rtol=2e-9)
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_contraction_keeps_native_derivatives(dtype, double_precision):
    product = o3.TensorProduct("2x4e", "2x4e", "2x4e", [(0, 0, 0, "uuu", False)])
    linear = o3.Linear("2x4e", "2x4e", internal_weights=False, shared_weights=False)
    module = UuuFullyConnectedSymmetricContraction(
        [product, product], [linear] * 3, backend="cuda"
    ).to(device="cuda", dtype=dtype)
    types = torch.tensor([0, 1, 0], device="cuda")
    values = tuple(
        (torch.randn(shape, device="cuda", dtype=dtype) * 0.2).requires_grad_()
        for shape in ((3, 18), (2, 4), (2, 4), (2, 4))
    )

    def joint(x, *weights):
        energy = module(x, weights, types).sin().sum()
        force = torch.autograd.grad(energy, x, create_graph=True)[0]
        gradients = torch.autograd.grad(energy + force.square().sum(), (x, *weights))
        return energy, force, *gradients

    graph = make_fx(joint)(*values)
    assert "eqx.ace_contract" in graph.code
    depths = {}
    for node in graph.graph.nodes:
        depths[node] = 1 + max((depths[x] for x in node.all_input_nodes), default=0)
    assert max(depths.values()) < 300
    compiled = torch.compile(graph, fullgraph=True)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    for scale in (1.0, 0.7):
        inputs = tuple((value.detach() * scale).requires_grad_() for value in values)
        with torch.no_grad():
            actual = compiled(*inputs)
        module.backend = "torch"
        expected = joint(*inputs)
        module.backend = "cuda"
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
