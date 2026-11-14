"""Analytic STF projection, arbitrary-order AD and joint compilation."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3
from equivariantx.ops import config
from equivariantx.ops.torch.co3.basis import RankTwo, rank_two


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 7])
def test_rank_two(device, dtype, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    value = torch.randn(rows, 2, 9, 2, device=device, dtype=dtype)[..., 0]
    matrix = co3.Projector.projection_matrix(2).to(value)
    torch.testing.assert_close(rank_two(value), value @ matrix)
    torch.testing.assert_close(RankTwo.apply(value), value @ matrix)


@pytest.mark.parametrize("function", [rank_two, RankTwo.apply])
def test_high_order(function):
    value = torch.randn(3, 2, 9, dtype=torch.float64, requires_grad=True)
    matrix = co3.Projector.projection_matrix(2)
    losses = [x.sin().sum() for x in (function(value), value @ matrix)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, value, create_graph=True)[0] for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-12, rtol=3e-12)
        tangent = torch.randn_like(value) * 0.2
        losses = [(gradient * tangent).sum() for gradient in gradients]
    tangent = torch.randn_like(value)

    def reference(x):
        return x @ matrix

    for fn, ref in (
        (function, reference),
        (
            torch.func.grad(lambda x: function(x).sin().sum()),
            torch.func.grad(lambda x: reference(x).sin().sum()),
        ),
    ):
        torch.testing.assert_close(
            torch.func.jvp(fn, (value,), (tangent,)),
            torch.func.jvp(ref, (value,), (tangent,)),
        )
    torch.testing.assert_close(torch.vmap(function)(value), reference(value))


def test_reference_switch(monkeypatch):
    layer = co3.Projector("2x2e").double()
    value = torch.randn(5, 18, dtype=torch.float64)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    reference = make_fx(layer)(value)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    optimized = make_fx(layer)(value)
    assert any(
        node.target == torch.ops.aten.mm.default for node in reference.graph.nodes
    )
    assert not any(
        node.target == torch.ops.aten.mm.default for node in optimized.graph.nodes
    )
    torch.testing.assert_close(optimized(value), reference(value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler")
def test_fullgraph_compile():
    layer = co3.Projector("2x2e+1x0e+3x2e+1x3e").double().cuda()
    value = torch.randn(4, layer.irreps_in.dim, dtype=torch.float64, device="cuda")
    torch.testing.assert_close(
        torch.compile(layer, fullgraph=True)(value), layer(value)
    )


def test_change_basis_precision():
    from equivariantx.tests.utils.testing import check_method_precision

    layer = co3.ChangeOfBasis("2x2e+1x3e").double()
    value = torch.randn(5, layer.input_dim, dtype=torch.float64)
    layer.float().double()
    for rank in (2, 3):
        torch.testing.assert_close(
            getattr(layer, f"basis_{rank}"), co3.path_matrix(rank)
        )
    check_method_precision(layer, [("forward", (value,), {})])
    with pytest.raises(ValueError, match="backend"):
        co3.ChangeOfBasis("2e", backend="invalid")
    features = torch.randn(3, 5)
    torch.testing.assert_close(
        co3.ChangeOfBasis("2e", backend="cuda")(features),
        co3.ChangeOfBasis("2e", backend="torch")(features),
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("rows", [0, 5])
def test_change_basis_grouped(inverse, rows, monkeypatch):
    layer = co3.ChangeOfBasis("2x2e+3x2o+1x2e+1x1o+2x2e", inverse).double()
    value = torch.randn(rows, 2 * layer.input_dim, dtype=torch.float64)[:, ::2]
    value.requires_grad_()
    actual = layer(value)
    with monkeypatch.context() as flags:
        flags.setattr(config, "TORCH_REFERENCE", True)
        expected = layer(value)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        actual, expected = [
            torch.autograd.grad(y.sin().sum(), value, create_graph=True)[0] * 0.2
            for y in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)


def test_change_basis_dynamic_force_loss(monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    embed = co3.ChangeOfBasis("2x2e+1x2o+1x3e").double()
    extract = co3.ChangeOfBasis("2x2e+1x2o+1x3e", True).double()

    def joint(x, w):
        energy = extract(embed(x * w).sin()).square().sum() * 0.01
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = (force - 0.1).square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    weight = (torch.randn(embed.input_dim, dtype=torch.float64) * 0.2).requires_grad_()
    x = (torch.randn(7, embed.input_dim, dtype=torch.float64) * 0.2).requires_grad_()
    count = ShapeVar("nodes", min=2, optimization_hint=7)
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec({"x": TensorSpec([count, None]), "w": TensorSpec([None])})
        ),
    )(x, weight)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = []

    def backend(module, arguments):
        from torch._inductor.compile_fx import compile_fx

        compiled.append(module)
        return compile_fx(module, arguments)

    run = torch.compile(graph, backend=backend, dynamic=True, fullgraph=True)
    for nodes in (7, 11, 9):
        x = (
            torch.randn(nodes, embed.input_dim, dtype=torch.float64) * 0.2
        ).requires_grad_()
        for _ in range(3):
            with monkeypatch.context() as flags:
                flags.setattr(config, "TORCH_REFERENCE", True)
                expected = joint(x, weight)
            actual = run(x.detach(), weight.detach())
            torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
            weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
    assert len(compiled) == 1
