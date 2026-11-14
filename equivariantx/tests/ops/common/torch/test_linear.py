"""Indexed channel maps, transposed roles and bounded coefficient gradients."""

import os
import subprocess
import sys

import pytest
import torch

from equivariantx.ops import config, get_kernel
from equivariantx.ops.common.torch import linear


def linear_case(nodes, channels=3, experts=2, layout="mul_ir", device="cpu"):
    ci, co, dim, elements = channels, channels + 2, 3, 4
    din, dout = experts * (ci * dim + 2), experts * (co * dim + 2)
    paths = (
        (experts, 2, experts, ci, co, dim, 0.7),
        (experts, 2 + ci * co, experts, ci, co, dim, -0.2),
    )
    numel = 2 + 2 * ci * co
    metadata = repr((din, dout, numel, experts, paths, layout))
    values = tuple(
        (torch.randn(*shape, device=device) * 0.2).requires_grad_()
        for shape in ((nodes, din), (elements, experts, numel), (nodes, dout))
    )
    types = torch.arange(nodes, device=device) % 2
    return metadata, values, types


@pytest.mark.parametrize(
    "nodes,channels,elements,expected",
    [
        (7, 64, 16, False),
        (256, 32, 16, False),
        (256, 64, 16, True),
        (256, 64, 89, False),
    ],
)
def test_cpu_grouping_policy(nodes, channels, elements, expected, monkeypatch):
    features = torch.empty(nodes, channels)
    weight = torch.empty(elements, 1, channels * channels)
    paths = ((0, 0, 0, channels, channels, 1, 1.0),)
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", True)
    assert linear.grouped_linear_available(features, weight, paths) is expected
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", False)
    assert not linear.grouped_linear_available(features, weight, paths)


@pytest.mark.parametrize(
    "nodes,channels,experts,dtype,expected",
    [
        (512, 64, 1, torch.float32, True),
        (256, 64, 4, torch.float32, False),
        (257, 128, 8, torch.float32, True),
        (512, 128, 8, torch.float32, True),
        (257, 128, 8, torch.float64, False),
        (7, 128, 8, torch.float32, False),
        (256, 1024, 2, torch.float32, False),
    ],
)
def test_cpu_capacity_policy(nodes, channels, experts, dtype, expected, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    calls = []
    grouping = linear.element_grouping

    def record(*args, tight=False):
        calls.append(tight)
        return grouping(*args, tight=tight)

    monkeypatch.setattr(linear, "element_grouping", record)
    with FakeTensorMode():
        linear.element_linear_grouped(
            torch.empty(nodes, experts * channels, dtype=dtype),
            torch.empty(16, experts, channels**2, dtype=dtype),
            torch.empty(nodes, dtype=torch.int64),
            ((0, 0, 0, channels, channels, 1, 1.0),),
            experts * channels,
            "mul_ir",
        )
    assert calls == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA symbolic policy")
@pytest.mark.parametrize("channels", [64, 128])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_public_unbacked_cuda(channels, backend, monkeypatch):
    from e3nn import o3
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.o3 import ElementLinear

    description = o3.Linear(
        f"{channels}x0e+{channels}x1o",
        f"{channels}x0e+{channels}x1o",
        internal_weights=False,
        shared_weights=False,
    )
    module = ElementLinear(description, backend=backend)
    ordinary = ElementLinear(description, backend="torch")
    weight = (
        torch.randn(4, module.weight_numel, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()

    def data(nodes):
        return (
            (
                torch.randn(nodes, module.input_dim, device="cuda", dtype=torch.float64)
                * 0.1
            ).requires_grad_(),
            weight,
            torch.arange(nodes, device="cuda") % 4,
        )

    def joint(function, x, w, types):
        value = function(x, w, types)
        energy = 1 + value.sin().sum() / (x.shape[0] + 1)
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force - 0.01).square().sum()
        return energy, force, torch.autograd.grad(loss, w)[0]

    count = ShapeVar("nodes", min=2, max=1024, optimization_hint=7)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    graph = make_fx(
        lambda *args: joint(module, *args),
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "*args": [
                        TensorSpec([count, None]),
                        TensorSpec([None, None]),
                        TensorSpec([count]),
                    ]
                }
            )
        ),
    )(*data(7))
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled_graphs = []

    def compiler(module, inputs):
        from torch._inductor.compile_fx import compile_fx

        compiled_graphs.append(module)
        return compile_fx(module, inputs)

    run = torch.compile(graph, dynamic=True, fullgraph=True, backend=compiler)
    for nodes in (7, 11, 513):
        values = data(nodes)
        with torch.no_grad():
            weight.add_(0.001)
        inputs = tuple(value.detach() for value in values)
        torch._dynamo.mark_dynamic(inputs[0], 0, min=2, max=1024)
        torch._dynamo.mark_static(inputs[0], 1)
        torch._dynamo.mark_static(inputs[1])
        torch._dynamo.mark_dynamic(inputs[2], 0, min=2, max=1024)
        actual = run(*inputs)
        with monkeypatch.context() as flags:
            flags.setattr(config, "TORCH_REFERENCE", True)
            expected = joint(ordinary, *values)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    assert len(compiled_graphs) == 1


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("experts", [1, 2])
@pytest.mark.parametrize("nodes", [0, 7])
def test_grouped_transposes(layout, experts, nodes, double_precision):
    metadata, values, types = linear_case(nodes, experts=experts, layout=layout)
    for role in range(3):
        actual = linear.contract_grouped(metadata, role, types, *values)
        expected = linear.contract_reference(metadata, role, types, *values)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        active = tuple(value for i, value in enumerate(values) if i != role)
        directions = tuple(torch.randn_like(value) * 0.1 for value in active)
        actual, expected = (
            value.sin().sum() / max(value.numel(), 1) for value in (actual, expected)
        )
        for _ in range(4):
            gradients = [
                torch.autograd.grad(value, active, create_graph=True)
                for value in (actual, expected)
            ]
            torch.testing.assert_close(*gradients, atol=1e-12, rtol=1e-12)
            actual, expected = [
                sum((g * d).sum() for g, d in zip(gs, directions)) for gs in gradients
            ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA packing policy")
@pytest.mark.parametrize("channels", [64, 128])
@pytest.mark.parametrize("reference", [False, True])
def test_provider_dispatch(channels, reference, double_precision, monkeypatch):
    metadata, values, types = linear_case(2048, channels, device="cuda")
    grouped = linear.contract_grouped
    calls = []

    def record(*args):
        calls.append(args[1])
        return grouped(*args)

    monkeypatch.setattr(linear, "contract_grouped", record)
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    for role in range(3):
        actual = get_kernel("element_linear")(metadata, role, types, *values)
        expected = linear.contract_reference(metadata, role, types, *values)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert calls == ([0, 1, 2] if channels == 128 and not reference else [])


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_grouped_functional_compilation(role, layout, double_precision):
    from equivariantx.tests.ops.testing import assert_functional_derivatives

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata, values, types = linear_case(7, layout=layout, device=device)

    def function(*args):
        return linear.contract_grouped(metadata, role, args[-1], *args[:-1])

    def reference(*args):
        return linear.contract_reference(metadata, role, args[-1], *args[:-1])

    assert_functional_derivatives(
        function, reference, (*values, types), compiler="inductor"
    )


@pytest.mark.parametrize("role", [0, 1])
def test_dynamic_transposes(role, double_precision):
    metadata = linear_case(7)[0]
    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def function(*args):
        return linear.contract_grouped(metadata, role, args[-1], *args[:-1])

    compiled = torch.compile(function, backend=compiler, dynamic=True, fullgraph=True)
    for nodes in (7, 11, 19):
        _, values, types = linear_case(nodes)
        for value in (values[0], values[2], types):
            torch._dynamo.mark_dynamic(value, 0, min=2, max=32)
        torch.testing.assert_close(
            compiled(*values, types),
            linear.contract_reference(metadata, role, types, *values),
            atol=1e-12,
            rtol=1e-12,
        )
    assert len(graphs) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
def test_grouped_force_aoti(tmp_path, double_precision):
    from e3nn import o3
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import o3 as eqx_o3
    from equivariantx.ops.common.linear import contract
    from equivariantx.ops.decompositions import polynomial_decompositions

    module = eqx_o3.MoEElementLinear(
        o3.Linear(
            "128x0e+128x1o",
            "130x0e+130x1o",
            internal_weights=False,
            shared_weights=False,
        ),
        2,
        backend="cuda",
    )
    metadata = module.metadata
    values = tuple(
        (torch.randn(*shape, device="cuda") * 0.2).requires_grad_()
        for shape in ((2048, module.input_dim), (4, 2, module.weight_numel))
    )
    types = torch.arange(2048, device="cuda") % 2

    def joint(function, x, weight, indices):
        output = function(
            metadata,
            2,
            indices,
            x,
            weight,
            x.new_empty((x.shape[0], module.output_dim)),
        )
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy + (force - 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    raw = make_fx(lambda *inputs: joint(contract, *inputs))(*values[:2], types)
    inputs = (*[value.detach() for value in values[:2]], types)
    graph = make_fx(raw, decomposition_table=polynomial_decompositions())(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(graph, inputs)
    package = tmp_path / "linear.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale, indices in ((1.0, types), (0.7, torch.zeros_like(types))):
        changed = tuple(
            (value.detach() * scale).requires_grad_() for value in values[:2]
        )
        cases.append(
            {
                "inputs": (*[value.detach().cpu() for value in changed], indices.cpu()),
                "expected": tuple(
                    value.detach().cpu()
                    for value in joint(linear.contract_reference, *changed, indices)
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


@pytest.mark.parametrize("elements", [1, 3])
@pytest.mark.parametrize("active", [0, 1])
def test_shared_coefficients_partial_ad(elements, active, double_precision):
    values = (torch.randn(elements, 5) * 0.1, torch.randn(5) * 0.1)
    values = tuple(
        value.requires_grad_(index == active) for index, value in enumerate(values)
    )
    direction = torch.randn_like(values[active]) * 0.1

    def function(value):
        arguments = list(values)
        arguments[active] = value
        return linear.SharedCoefficients.apply(*arguments)

    def reference(value):
        arguments = list(values)
        arguments[active] = value
        return arguments[0] + arguments[1]

    actual = torch.func.jvp(function, (values[active],), (direction,))
    expected = torch.func.jvp(reference, (values[active],), (direction,))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(values[active], direction)
        actual = torch.autograd.forward_ad.unpack_dual(function(dual))
        expected = torch.autograd.forward_ad.unpack_dual(reference(dual))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    actual, expected = [
        call(values[active]).sin().sum() for call in (function, reference)
    ]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(value, values[active], create_graph=True)[0]
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-12, rtol=1e-12)
        actual, expected = [(value * direction).sum() for value in gradients]
    function(values[active]).square().sum().backward()
    expected = torch.autograd.grad(
        reference(values[active]).square().sum(), values[active]
    )[0]
    torch.testing.assert_close(values[active].grad, expected, atol=1e-12, rtol=1e-12)
    assert values[1 - active].grad is None


@pytest.mark.parametrize(
    "nodes,ci,co,experts,elements,device,enabled,reference,expected",
    [
        (256, 64, 64, 2, 16, "cpu", True, False, True),
        (256, 128, 128, 4, 16, "cpu", True, False, True),
        (512, 64, 64, 8, 16, "cpu", True, False, True),
        (511, 64, 64, 2, 16, "cpu", True, False, True),
        (257, 128, 128, 8, 16, "cpu", True, False, True),
        (257, 64, 64, 8, 16, "cpu", True, False, False),
        (7, 64, 64, 4, 16, "cpu", True, False, False),
        (256, 64, 64, 1, 16, "cpu", True, False, False),
        (256, 64, 64, 3, 16, "cpu", True, False, False),
        (256, 1024, 1024, 2, 16, "cpu", True, False, False),
        (256, 64, 128, 4, 16, "cpu", True, False, False),
        (256, 64, 64, 4, 64, "cpu", True, False, False),
        (1024, 64, 64, 4, 16, "cpu", True, False, False),
        (256, 64, 64, 4, 16, "cpu", False, False, False),
        (256, 64, 64, 4, 16, "cpu", True, True, False),
        (2048, 64, 64, 4, 16, "cuda", True, False, False),
    ],
)
def test_expert_grouping_policy(
    nodes, ci, co, experts, elements, device, enabled, reference, expected, monkeypatch
):
    from torch.fx.experimental.proxy_tensor import make_fx

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    features = torch.empty(nodes, experts * ci, device=device)
    weight = torch.empty(elements, experts, ci * co, device=device)
    indices = torch.arange(nodes, device=device) % elements
    paths = ((0, 0, 0, ci, co, 1, 1.0),)
    calls = []
    implementation = linear.element_expert_path

    def record(*args):
        calls.append(args[-1])
        return implementation(*args)

    monkeypatch.setattr(linear, "element_expert_path", record)
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", enabled)
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    make_fx(
        lambda x, w, types: linear.element_linear(
            x, w, types, paths, experts * co, "mul_ir"
        ),
        tracing_mode="fake",
    )(features, weight, indices)
    assert bool(calls) is expected


def test_expert_symbolic_fallback(double_precision, monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    paths = ((0, 0, 0, 64, 64, 1, 0.125),)
    weight = (torch.randn(16, 4, 4096) * 0.1).requires_grad_()

    def data(nodes):
        return (
            (torch.randn(nodes, 256) * 0.1).requires_grad_(),
            weight,
            torch.arange(nodes) % 16,
        )

    def forbidden(*args):
        raise AssertionError("Symbolic node counts must retain the existing path")

    def joint(function, x, w, types):
        output = function(x, w, types, paths, 256, "mul_ir")
        energy = output.sin().sum() / (x.shape[0] + 1)
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return output, force, torch.autograd.grad(loss, w)[0]

    monkeypatch.setattr(linear, "element_expert_path", forbidden)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", True)
    nodes = ShapeVar("nodes", min=2, max=1024, optimization_hint=7)
    graph = make_fx(
        lambda *args: joint(linear.element_linear, *args),
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "*args": [
                        TensorSpec((nodes, None)),
                        TensorSpec((None, None, None)),
                        TensorSpec((nodes,)),
                    ]
                }
            )
        ),
    )(*data(7))
    graphs = []

    def compiler(module, inputs):
        graphs.append(module)
        return module.forward

    compiled = torch.compile(graph, backend=compiler, dynamic=True, fullgraph=True)
    for count in (7, 11, 257):
        values = data(count)
        inputs = tuple(value.detach() for value in values)
        for value in (inputs[0], inputs[2]):
            torch._dynamo.mark_dynamic(value, 0, min=2, max=1024)
        torch._dynamo.mark_static(inputs[0], 1)
        torch._dynamo.mark_static(inputs[1])
        actual = compiled(*inputs)
        expected = joint(linear.element_linear_reference, *values)
        torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)
    assert len(graphs) == 1


def test_expert_dynamo_fallback(double_precision, monkeypatch):
    paths = ((0, 0, 0, 64, 64, 3, 0.125),)
    weight = (torch.randn(4096, 16, 4).permute(1, 2, 0) * 0.1).requires_grad_()
    graphs, arguments = [], []
    first = None

    def forbidden(*args):
        raise AssertionError("Dynamo symbolic sizes must not specialize the schedule")

    def compiler(module, inputs):
        graphs.append(module)

        def execute(*args):
            arguments.append(args)
            return module(*args)

        return execute

    monkeypatch.setattr(linear, "element_expert_path", forbidden)
    compiled = torch.compile(
        lambda x, w, types: linear.element_linear_grouped(
            x, w, types, paths, 768, "mul_ir"
        ),
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    for nodes in (7, 11, 257):
        x = (torch.randn(768, nodes).T * 0.1).requires_grad_()
        types = (torch.arange(2 * nodes) % 16)[::2]
        for value in (x, types):
            torch._dynamo.mark_dynamic(value, 0, min=2, max=1024)
        torch._dynamo.mark_static(x, 1)
        torch._dynamo.mark_static(weight)
        saved_inputs = x.detach().clone(), weight.detach().clone()
        actual = compiled(x, weight, types)
        expected = linear.element_linear_reference(
            x, weight, types, paths, 768, "mul_ir"
        )
        torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
        if first is None:
            args = arguments[-1]
            first = (
                args,
                next(i for i, value in enumerate(args) if value is x),
                next(i for i, value in enumerate(args) if value is weight),
                types,
                saved_inputs,
            )
        prediction = actual
        saved_prediction = prediction.detach().clone()
        directions = torch.randn_like(x) * 0.1, torch.randn_like(weight) * 0.1
        actual, expected = [
            value.sin().sum() / (nodes + 1) for value in (actual, expected)
        ]
        for _ in range(3):
            gradients = [
                torch.autograd.grad(value, (x, weight), create_graph=True)
                for value in (actual, expected)
            ]
            torch.testing.assert_close(*gradients, atol=2e-9, rtol=2e-9)
            actual, expected = [
                sum(
                    (value * direction).sum()
                    for value, direction in zip(gradient, directions)
                )
                for gradient in gradients
            ]
        gradient = gradients[0][1]
        assert gradient.untyped_storage().data_ptr() not in {
            value.untyped_storage().data_ptr() for value in (x, weight, prediction)
        }
        weight.grad = gradient
        torch.nn.utils.clip_grad_norm_([weight], 0.05, foreach=True)
        torch.testing.assert_close((x, weight), saved_inputs, atol=0, rtol=0)
        torch.testing.assert_close(prediction, saved_prediction, atol=0, rtol=0)
    assert len(graphs) == 1

    args, x_position, w_position, types, primals = first

    def traced(x, w):
        values = list(args)
        values[x_position], values[w_position] = x, w
        return graphs[0](*values)[0]

    def reference(x, w):
        return linear.element_linear_reference(x, w, types, paths, 768, "mul_ir")

    primals = tuple(torch.stack((value, value * 0.7)) for value in primals)
    tangents = tuple(torch.randn_like(value) * 0.1 for value in primals)

    def transformed(function):
        return torch.vmap(
            lambda x, w, dx, dw: torch.func.jvp(function, (x, w), (dx, dw))
        )(*primals, *tangents)

    torch.testing.assert_close(
        transformed(traced), transformed(reference), atol=5e-11, rtol=5e-11
    )


def test_expert_symbolic_metadata(double_precision, monkeypatch):
    paths = ((0, 0, 0, 64, 64, 1, 0.125),)
    graphs = []

    def forbidden(*args):
        raise AssertionError("Unknown expert counts must retain the existing path")

    def compiler(module, inputs):
        graphs.append(module)
        return module.forward

    monkeypatch.setattr(linear, "element_expert_path", forbidden)
    compiled = torch.compile(
        lambda x, w, types: linear.element_linear_grouped(
            x, w, types, paths, w.shape[1] * 64, "mul_ir"
        ),
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    for experts in (2, 4, 8):
        x = (torch.randn(256, experts * 64) * 0.1).requires_grad_()
        weight = (torch.randn(16, experts, 4096) * 0.1).requires_grad_()
        types = torch.arange(256) % 16
        torch._dynamo.mark_static(x, 0)
        torch._dynamo.mark_static(types)
        torch._dynamo.mark_static(weight, 0)
        torch._dynamo.mark_static(weight, 2)
        torch._dynamo.mark_dynamic(weight, 1, min=2, max=8)
        actual = compiled(x, weight, types)
        expected = linear.element_linear_reference(
            x, weight, types, paths, experts * 64, "mul_ir"
        )
        torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    assert len(graphs) == 1


@pytest.mark.parametrize(
    "lower,elements,channels,device,expected",
    [
        (256, 16, 64, "cpu", True),
        (2, 16, 64, "cpu", False),
        (128, 16, 64, "cpu", False),
        (256, 89, 64, "cpu", False),
        (356, 89, 64, "cpu", True),
        (256, 16, 32, "cpu", False),
        (256, 16, 1024, "cpu", True),
        pytest.param(
            256,
            16,
            64,
            "cuda",
            False,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA symbolic policy"
            ),
        ),
    ],
)
def test_dynamic_grouping_permission(
    lower, elements, channels, device, expected, monkeypatch
):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", True)
    paths = ((0, 0, 0, channels, channels, 1, 1.0),)
    inputs = (
        torch.empty(512, channels, device=device),
        torch.empty(elements, 1, channels * channels, device=device),
        torch.arange(512, device=device) % elements,
    )

    def operation(x, w, types):
        assert not linear.grouped_linear_available(x, w, paths)
        assert linear.grouped_linear_available(x, w, paths, dynamic=True) is expected
        return linear.element_linear(x, w, types, paths, channels, "mul_ir")

    count = ShapeVar("nodes", min=lower, max=1024, optimization_hint=512)
    graph = make_fx(
        operation,
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec((count, None)),
                    "w": TensorSpec((None, None, None)),
                    "types": TensorSpec((count,)),
                }
            )
        ),
    )(*inputs)
    assert (
        any(node.target == torch.ops.aten.sort.default for node in graph.graph.nodes)
        is expected
    )


@pytest.mark.parametrize("role", [0, 1, 2])
def test_dynamic_contract_keeps_reference(role, monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    paths = ((0, 0, 0, 1024, 1024, 1, 1.0),)
    metadata = repr((1024, 1024, 1024**2, 1, paths, "mul_ir"))
    inputs = (
        torch.empty(256, 1024),
        torch.empty(16, 1, 1024**2),
        torch.empty(256, 1024),
        torch.zeros(256, dtype=torch.int64),
    )

    def forbidden(*args):
        raise AssertionError("Only the element_linear primal permits dynamic grouping")

    monkeypatch.setattr(linear, "contract_grouped", forbidden)
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)

    def capture(function):
        count = ShapeVar("nodes", min=256, max=1024, optimization_hint=256)
        return make_fx(
            lambda *args: function(metadata, role, args[3], *args[:3]),
            tracing_mode="fake",
            dynamic_shapes=ShapesSpec(
                ParamsSpec(
                    {
                        "*args": [
                            TensorSpec((count, None)),
                            TensorSpec((None, None, None)),
                            TensorSpec((count, None)),
                            TensorSpec((count,)),
                        ]
                    }
                )
            ),
        )(*inputs)

    actual, expected = capture(linear.contract), capture(linear.contract_reference)
    assert actual.code == expected.code
