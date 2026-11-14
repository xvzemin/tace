"""Planar signed projections and independent matrix derivatives."""

import pytest
import torch
from torch.fx.experimental.dynamic_spec import (
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config
from equivariantx.ops.torch.co2.basis import planar_project
from equivariantx.tests.utils.testing import check_forward_ad


def test_change_basis_backend():
    with pytest.raises(ValueError, match="backend"):
        co2.ChangeOfBasis("1m", backend="unsupported")
    features = torch.randn(3, 2)
    torch.testing.assert_close(
        co2.ChangeOfBasis("1m", backend="cuda")(features),
        co2.ChangeOfBasis("1m", backend="torch")(features),
    )


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 7])
def test_planar_projection(rank, dtype, rows):
    value = torch.randn(rows, 3, 2**rank, 2, dtype=dtype)[..., 0]
    matrix = co2.projector_matrix(rank).to(value)
    torch.testing.assert_close(planar_project(value, rank), value @ matrix)


@pytest.mark.parametrize("rank", [2, 4, 6])
def test_high_order_and_transforms(rank):
    x = (torch.randn(3, 2, 2**rank, dtype=torch.float64) * 0.2).requires_grad_()
    matrix = co2.projector_matrix(rank)

    def actual(x):
        return planar_project(x, rank)

    def reference(x):
        return x @ matrix

    values = actual(x), reference(x)
    for _ in range(5):
        torch.testing.assert_close(*values, atol=2e-11, rtol=2e-11)
        direction = torch.randn_like(x) * 0.1
        values = [
            torch.autograd.grad(value.sin().sum(), x, create_graph=True)[0] * direction
            for value in values
        ]
    torch.testing.assert_close(*values, atol=2e-11, rtol=2e-11)
    check_forward_ad(actual, reference, (x,))
    torch.testing.assert_close(torch.vmap(actual)(x), reference(x))
    assert torch.autograd.gradgradcheck(actual, (x,), fast_mode=True)


def test_reference_and_rank_policy(monkeypatch):
    layer = co2.Projector("2x4m+1x0e+1x4m").double()
    x = torch.randn(7, layer.input_dim, dtype=torch.float64)
    monkeypatch.setattr(config, "TORCH_PLANAR_PROJECTION", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    optimized = make_fx(layer)(x)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    reference = make_fx(layer)(x)
    assert not any(n.target == torch.ops.aten.mm.default for n in optimized.graph.nodes)
    assert any(n.target == torch.ops.aten.mm.default for n in reference.graph.nodes)
    torch.testing.assert_close(optimized(x), reference(x), atol=2e-12, rtol=2e-12)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    for rank, method in ((2, "matrix"), (3, "matrix"), (4, "factorized")):
        layer = co2.Projector(f"2x{rank}m", method=method).double()
        graph = make_fx(layer)(torch.randn(7, layer.input_dim, dtype=torch.float64))
        assert any(n.target == torch.ops.aten.mm.default for n in graph.graph.nodes)
    monkeypatch.setattr(config, "TORCH_PLANAR_PROJECTION", False)
    layer = co2.Projector("2x4m").double()
    graph = make_fx(layer)(torch.randn(7, layer.input_dim, dtype=torch.float64))
    assert any(n.target == torch.ops.aten.mm.default for n in graph.graph.nodes)


def test_direct_dynamic_compile(monkeypatch):
    layer = co2.Projector("2x4m+1x0e+1x4m").double()
    run = torch.compile(layer, fullgraph=True, dynamic=True)
    for rows in (3, 7, 0):
        x = torch.randn(rows, layer.input_dim, dtype=torch.float64)
        actual = run(x)
        with monkeypatch.context() as flags:
            flags.setattr(config, "TORCH_REFERENCE", True)
            expected = layer(x)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("rank", [4, 6])
def test_dynamic_force_loss(rank, monkeypatch):
    layer = co2.Projector(f"2x{rank}m").double()
    weight = (torch.randn(layer.input_dim, dtype=torch.float64) * 0.2).requires_grad_()

    def joint(x, w):
        energy = layer(x * w).sin().sum() * 0.1
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, torch.autograd.grad(loss, w)[0]

    count = ShapeVar("nodes", min=2, optimization_hint=7)
    x = (torch.randn(7, layer.input_dim, dtype=torch.float64) * 0.2).requires_grad_()
    graph = make_fx(
        joint,
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(
            ParamsSpec({"x": TensorSpec([count, None]), "w": TensorSpec([None])})
        ),
    )(x, weight)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    graphs = []

    def compiler(module, inputs):
        from torch._inductor.compile_fx import compile_fx

        graphs.append(module)
        return compile_fx(module, inputs)

    run = torch.compile(graph, backend=compiler, fullgraph=True, dynamic=True)
    for rows in (7, 11, 9):
        x = (
            torch.randn(rows, layer.input_dim, dtype=torch.float64) * 0.2
        ).requires_grad_()
        inputs = x.detach(), weight.detach()
        torch._dynamo.mark_dynamic(inputs[0], 0, min=2)
        torch._dynamo.mark_static(inputs[0], 1)
        torch._dynamo.mark_static(inputs[1])
        for _ in range(3):
            actual = run(*inputs)
            with monkeypatch.context() as flags:
                flags.setattr(config, "TORCH_REFERENCE", True)
                expected = joint(x, weight)
            torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
            with torch.no_grad():
                weight.sub_(expected[-1] * 0.01)
    assert len(graphs) == 1
