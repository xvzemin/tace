"""Bounded element packing, functional transforms and mixed derivatives."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.torch.grouped import (
    element_expert_path,
    element_grouping,
    element_path,
)
from equivariantx.ops.common.torch.linear import (
    element_linear_blocks,
    element_linear_grouped,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture(autouse=True)
def exact_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def grouped_case(nodes, experts, layout, device, dtype=torch.float64):
    # Duplicate output paths, non-aligned widths and unused output channels.
    paths = (
        (0, 0, 0, 5, 7, 3, 0.5),
        (15 * experts, 35, 0, 4, 7, 3, 0.25),
        (27 * experts, 63, 24 * experts, 3, 2, 1, 0.75),
    )
    x = (
        torch.randn(30 * experts, nodes, device=device, dtype=dtype).T * 0.2
    ).requires_grad_()
    w = (torch.randn(5, experts, 69, device=device, dtype=dtype) * 0.2).requires_grad_()
    types = torch.arange(nodes, device=device) % (1 if nodes > 100 else 3)

    def optimized(x, w, types):
        return element_linear_grouped(x, w, types, paths, 28 * experts, layout)

    def reference(x, w, types):
        return element_linear_blocks(
            x.unsqueeze(1), w.index_select(0, types), paths, 28 * experts, layout
        ).squeeze(1)

    return optimized, reference, (x, w, types)


@pytest.mark.parametrize("nodes", [0, 7, 257])
@pytest.mark.parametrize("experts", [1, 2])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_element_derivatives(nodes, experts, layout, device, dtype):
    optimized, reference, (x, w, types) = grouped_case(
        nodes, experts, layout, device, dtype
    )
    tolerance = 5e-6 if dtype == torch.float32 else 1e-11
    actual, expected = optimized(x, w, types), reference(x, w, types)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for _ in range(4):
        seed = torch.randn_like(actual) * 0.2
        gradients = [
            torch.autograd.grad((y.sin() * seed).sum(), (x, w), create_graph=True)
            for y in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=tolerance, rtol=tolerance)
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]
    tangents = torch.randn_like(x), torch.randn_like(w)
    torch.testing.assert_close(
        torch.func.jvp(lambda x, w: optimized(x, w, types), (x, w), tangents),
        torch.func.jvp(lambda x, w: reference(x, w, types), (x, w), tangents),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_vmap(layout, device):
    optimized, reference, (x, w, types) = grouped_case(7, 2, layout, device)
    values = (
        x.detach().repeat(2, 1, 1),
        w.detach().repeat(2, 1, 1, 1),
        torch.stack((types, types.roll(1))),
    )
    torch.testing.assert_close(
        torch.vmap(optimized)(*values),
        torch.vmap(reference)(*values),
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_joint_compilation(layout, device):
    optimized, reference, inputs = grouped_case(7, 2, layout, device)

    def joint(function, x, w, types):
        output = function(x, w, types)
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        gradient = torch.autograd.grad((force - 0.1).square().mean() + energy, w)[0]
        return energy, force, gradient

    graph = make_fx(lambda *values: joint(optimized, *values))(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    compiled = torch.compile(graph, fullgraph=True)
    detached = tuple(value.detach() for value in inputs)
    with torch.no_grad():
        actual = compiled(*detached)
    expected = joint(reference, *inputs)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_dynamic_groups(device):
    optimized, reference, inputs = grouped_case(7, 2, "mul_ir", device)
    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(optimized, backend=compiler, dynamic=True, fullgraph=True)
    for nodes in (7, 11, 19):
        _, _, (x, w, types) = grouped_case(nodes, 2, "mul_ir", device)
        torch._dynamo.mark_dynamic(x, 0, min=2)
        torch._dynamo.mark_dynamic(types, 0, min=2)
        torch.testing.assert_close(compiled(x, w, types), reference(x, w, types))
    assert len(graphs) == 1


@pytest.mark.parametrize("experts", [1, 2])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_symbolic_empty_force_graph(experts, layout):
    optimized, reference, inputs = grouped_case(7, experts, layout, "cpu")

    def joint(function, x, w, types):
        output = function(x, w, types)
        energy = output.sin().sum() / (x.shape[0] + 1)
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force - 0.1).square().sum()
        return force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(lambda *args: joint(optimized, *args), tracing_mode="symbolic")(
        *inputs
    )
    assert not any(
        node.target == torch.ops.aten.sym_numel.default for node in graph.graph.nodes
    )
    compiled = torch.compile(graph, dynamic=True, fullgraph=True)
    for nodes in (7, 11, 5, 0):
        _, _, values = grouped_case(nodes, experts, layout, "cpu")
        actual = compiled(*(value.detach() for value in values))
        expected = joint(reference, *values)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("experts", [1, 3])
def test_int32_indices(experts):
    optimized, reference, (x, w, types) = grouped_case(13, experts, "mul_ir", "cpu")
    actual = optimized(x, w, types.to(torch.int32))
    expected = reference(x, w, types)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("experts,width", [(1, 1), (2, 4), (3, 8)])
def test_tight_capacity(experts, width):
    from itertools import product

    for nodes in range(8):
        elements = 3
        maximum = 0
        for counts in product(range(nodes + 1), repeat=elements):
            if sum(counts) != nodes:
                continue
            types = torch.repeat_interleave(
                torch.arange(elements), torch.tensor(counts)
            )
            old = element_grouping(types, elements, experts, width)
            slots, owners, chunks = element_grouping(
                types, elements, experts, width, tight=True
            )
            occupied = experts * sum((count + width - 1) // width for count in counts)
            maximum = max(maximum, occupied)
            assert occupied <= chunks <= old[2]
            torch.testing.assert_close(slots, old[0])
            torch.testing.assert_close(owners, old[1][:chunks])
            if nodes:
                assert slots.min() >= 0 and slots.max() < chunks * width
                negative = element_grouping(
                    (types - elements).to(torch.int32),
                    elements,
                    experts,
                    width,
                    tight=True,
                )
                torch.testing.assert_close((slots, owners), negative[:2])
        assert chunks == maximum


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_tight_derivatives(layout):
    x = (torch.randn(18, 7, dtype=torch.float64).T * 0.1).requires_grad_()
    w = (torch.randn(5, 2, 15, dtype=torch.float64) * 0.1).requires_grad_()
    types = torch.tensor([0, -1, 0, 2, -1, 2, 0])
    path = (0, 0, 0, 3, 5, 3, 0.5)

    def function(x, w, types, tight):
        groups = element_grouping(types, 5, 2, 4, tight=tight)
        return element_path(x, w, groups, path, layout, 4)

    actual, expected = [function(x, w, types, tight) for tight in (True, False)]
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    directions = torch.randn_like(x) * 0.1, torch.randn_like(w) * 0.1
    actual, expected = [
        (value.sin().sum() + value.square().mean()) for value in (actual, expected)
    ]
    for _ in range(5):
        derivatives = [
            torch.autograd.grad(value, (x, w), create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*derivatives, atol=2e-11, rtol=2e-11)
        actual, expected = [
            sum((g * d).sum() for g, d in zip(gradient, directions))
            for gradient in derivatives
        ]

    def transformed(x, w, types, tight):
        return torch.func.jvp(
            lambda x, w: function(x, w, types, tight), (x, w), directions
        )

    arguments = (
        x.repeat(2, 1, 1),
        w.repeat(2, 1, 1, 1),
        torch.stack((types, types.roll(1))),
    )
    torch.testing.assert_close(
        torch.vmap(lambda *args: transformed(*args, True))(*arguments),
        torch.vmap(lambda *args: transformed(*args, False))(*arguments),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
def test_element_force_aoti(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    from e3nn import o3

    from equivariantx import o3 as eqx_o3
    from equivariantx.ops import config

    linear = o3.Linear(
        "128x0e+128x1o", "128x0e+128x1o", internal_weights=False, shared_weights=False
    )
    module = eqx_o3.ElementLinear(linear)
    x = torch.randn(
        512, module.input_dim, device="cuda", dtype=torch.float64, requires_grad=True
    )
    w = torch.randn(
        4, module.weight_numel, device="cuda", dtype=x.dtype, requires_grad=True
    )
    types = torch.arange(512, device="cuda") % 3

    def joint(x, w, types):
        energy = module(x, w, types).square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        gradient = torch.autograd.grad(energy + (force - 0.1).square().sum(), w)[0]
        return energy, force, gradient

    inputs = x, w, types
    graph = make_fx(joint)(*inputs)
    assert any(
        node.target == torch.ops.aten.searchsorted.Tensor for node in graph.graph.nodes
    )
    exported = torch.export.export(graph, tuple(value.detach() for value in inputs))
    package = tmp_path / "element.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for scale in (1.0, 0.8):
        values = (
            x,
            (w.detach() * scale).requires_grad_(),
            types if scale == 1.0 else torch.zeros_like(types),
        )
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in values),
                "expected": tuple(value.detach().cpu() for value in joint(*values)),
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
        torch.testing.assert_close(value.cpu(), expected, atol=1e-10, rtol=1e-10)
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


@pytest.mark.parametrize(
    "nodes,experts,layout,dtype",
    [
        (0, 2, "mul_ir", torch.float32),
        (1, 4, "ir_mul", torch.float64),
        (11, 8, "mul_ir", torch.float64),
        (11, 2, "ir_mul", torch.float32),
        (11, 4, "mul_ir", torch.float32),
        (11, 8, "ir_mul", torch.float64),
    ],
)
def test_expert_path_derivatives(nodes, experts, layout, dtype, device):
    ci, co, dim = 3, 5, 2
    features = (
        torch.randn(experts * (ci * dim + 2), nodes, dtype=dtype, device=device).T * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(5, experts, 2 * (ci * co + 5), dtype=dtype, device=device) * 0.1
    )[..., ::2].requires_grad_()
    types = torch.arange(nodes, device=device) % 3
    path = (experts, 2, 0, ci, co, dim, 0.5)
    grouping = element_grouping(types, 5, 1, 8)

    def optimized(x, w):
        return element_expert_path(x, w, grouping, path, layout, 8)

    def reference(x, w):
        x = x[:, experts : experts * (ci * dim + 1)]
        if layout == "mul_ir":
            x = x.reshape(nodes, experts, ci, dim)
        else:
            x = x.reshape(nodes, dim, experts, ci).permute(0, 2, 3, 1)
        w = w[types, :, 2 : 2 + ci * co].reshape(nodes, experts, ci, co)
        y = torch.einsum("neid,neio->neod", x, w) * 0.5
        if layout == "ir_mul":
            y = y.permute(0, 3, 1, 2)
        return y.reshape(nodes, experts * co * dim)

    tolerance = 5e-6 if dtype == torch.float32 else 2e-11
    actual, expected = optimized(features, weight), reference(features, weight)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for _ in range(5):
        seed = torch.randn_like(actual) * 0.1
        gradients = [
            torch.autograd.grad(
                (value.sin() * seed).sum(), (features, weight), create_graph=True
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=tolerance, rtol=tolerance)
        actual, expected = [
            torch.cat([value.flatten() for value in gradient]) for gradient in gradients
        ]
    tangents = torch.randn_like(features), torch.randn_like(weight)
    torch.testing.assert_close(
        torch.func.jvp(optimized, (features, weight), tangents),
        torch.func.jvp(reference, (features, weight), tangents),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_unbacked_views_transforms(layout):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    optimized, reference, inputs = grouped_case(7, 2, layout, "cpu")
    count = ShapeVar("nodes", min=2, max=64, optimization_hint=7)
    graph = make_fx(
        optimized,
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
    assert any(
        node.target == torch.ops.aten.as_strided.default for node in graph.graph.nodes
    )
    for nodes in (7, 11, 19):
        _, _, (features, weight, types) = grouped_case(nodes, 2, layout, "cpu")
        weight = weight.detach().repeat_interleave(2, dim=-1)[..., ::2].requires_grad_()
        actual = graph(features, weight, types)
        expected = reference(features, weight, types)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        for _ in range(5):
            seed = torch.randn_like(actual) * 0.1
            gradients = [
                torch.autograd.grad(
                    (value.sin() * seed).sum(),
                    (features, weight),
                    create_graph=True,
                )
                for value in (actual, expected)
            ]
            torch.testing.assert_close(*gradients, atol=2e-11, rtol=2e-11)
            actual, expected = [
                torch.cat([value.flatten() for value in gradient])
                for gradient in gradients
            ]
        values = (
            features.detach().repeat(2, 1, 1),
            weight.detach().repeat(2, 1, 1, 1),
            types.repeat(2, 1),
        )
        tangents = torch.randn_like(values[0]), torch.randn_like(values[1])

        def transformed(function, x, w, t, dx, dw):
            return torch.func.jvp(lambda x, w: function(x, w, t), (x, w), (dx, dw))

        torch.testing.assert_close(
            torch.vmap(lambda *args: transformed(graph, *args))(*values, *tangents),
            torch.vmap(lambda *args: transformed(reference, *args))(*values, *tangents),
            atol=2e-11,
            rtol=2e-11,
        )
