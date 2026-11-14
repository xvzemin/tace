"""Bounded CPU contraction ordering and force-training gradients."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3
from equivariantx.ops import config
from equivariantx.ops.torch.co3.ictd import tensor_product as kernel


def mixed(channels, per_edge, dtype):
    return co3.ictd.TensorProduct(
        f"{channels}x1o+{channels}x1o+{channels}x0e",
        f"{channels}x2e+{channels}x2e+{channels}x1o",
        f"{channels}x3o+{channels}x3o+{channels}x0e+{channels}x2e",
        [
            (0, 0, 0, "uvw", True, 0.7),
            (0, 2, 3, "uvw", True, 0.4),
            (1, 0, 0, "uvw", True, 1.2),
            (0, 1, 1, "uvw", True, 0.8),
            (2, 1, 3, "uuw", True, 1.3),
            (0, 2, 2, "uuu", False, 0.2),
        ],
        internal_weights=False,
        shared_weights=not per_edge,
    ).to(dtype=dtype)


def inputs(layer, nodes, per_edge, dtype, freeze=False):
    values = []
    for index, width in enumerate((*layer.input_dims, layer.weight_numel)):
        shape = (nodes, width * 2) if index < 2 or per_edge else (width * 2,)
        value = (torch.randn(*shape, dtype=dtype) * 0.02)[..., ::2]
        values.append(value.requires_grad_(not freeze or index != 0))
    return tuple(values)


def force_loss(function, x, y, weight):
    output = function(x, y, weight)
    energy = output.sin().sum() * 0.01
    coordinates = (x,) if x is y else tuple(t for t in (x, y) if t.requires_grad)
    forces = torch.autograd.grad(energy, coordinates, create_graph=True)
    loss = sum((force - 0.01).square().sum() for force in forces)
    return output, *forces, torch.autograd.grad(loss, weight)[0]


def check_clipping(outputs, arguments):
    saved_inputs = tuple(value.detach().clone() for value in arguments)
    saved_outputs = tuple(value.detach().clone() for value in outputs[:-1])
    outputs[-1].mul_(0.125)
    torch.testing.assert_close(arguments, saved_inputs, atol=0, rtol=0)
    torch.testing.assert_close(outputs[:-1], saved_outputs, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nodes", [0, 7])
@pytest.mark.parametrize("per_edge", [False, True])
@pytest.mark.parametrize("freeze", [False, True])
def test_mixed_paths(dtype, nodes, per_edge, freeze, monkeypatch):
    monkeypatch.setattr(config, "TORCH_CARTESIAN_REORDER", True)
    monkeypatch.setattr(config, "CARTESIAN_REORDER_MIN_WORK", 4**3)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    layer = mixed(4, per_edge, dtype)
    values = inputs(layer, nodes, per_edge, dtype, freeze)
    graph = make_fx(layer)(*values)
    actual = force_loss(graph, *values)
    with monkeypatch.context() as flags:
        flags.setattr(config, "TORCH_REFERENCE", True)
        expected = force_loss(layer, *values)
    tolerance = 3e-6 if dtype == torch.float32 else 5e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    check_clipping(actual, values)


@pytest.mark.parametrize("per_edge", [False, True])
def test_self_coupling_and_forward_ad(per_edge, monkeypatch):
    monkeypatch.setattr(config, "TORCH_CARTESIAN_REORDER", True)
    monkeypatch.setattr(config, "CARTESIAN_REORDER_MIN_WORK", 4**3)
    layer = co3.ictd.TensorProduct(
        "4x1o",
        "4x1o",
        "4x2e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=not per_edge,
    ).double()
    x, _, w = inputs(layer, 7, per_edge, torch.float64)

    def reference(x, weight):
        return layer(x, x, weight)

    graph = make_fx(reference)(x, w)
    actual, expected = graph(x, w), reference(x, w)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
        actual, expected = [
            torch.cat(
                [
                    v.flatten()
                    for v in torch.autograd.grad(
                        y.sin().sum(), (x, w), create_graph=True
                    )
                ]
            )
            * 0.01
            for y in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    from equivariantx.tests.utils.testing import check_forward_ad

    check_forward_ad(graph, reference, (x, w))
    batched = torch.stack((x, x * 0.5))
    torch.testing.assert_close(
        torch.vmap(graph, in_dims=(0, None))(batched, w),
        reference(batched, w),
        atol=5e-11,
        rtol=5e-11,
    )
    actual = force_loss(lambda a, b, weight: graph(a, weight), x, x, w)
    expected = force_loss(layer, x, x, w)
    torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    check_clipping(actual, (x, x, w))


@pytest.mark.parametrize("per_edge", [False, True])
def test_broadcast_right_order(per_edge, monkeypatch):
    monkeypatch.setattr(config, "CARTESIAN_REORDER_MIN_WORK", 4**3)
    layer = co3.ictd.TensorProduct(
        "4x2e",
        "4x1o",
        "4x3o",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=not per_edge,
    ).double()
    assert kernel.channel_order(layer.groups[0]) == "right"
    x, y, weight = inputs(layer, 7, per_edge, torch.float64)
    x = torch.stack((x[:1], x[:1] * 0.5)).detach().requires_grad_()
    graph = make_fx(layer)(x, y, weight)
    actual = force_loss(graph, x, y, weight)
    with monkeypatch.context() as flags:
        flags.setattr(config, "TORCH_REFERENCE", True)
        expected = force_loss(layer, x, y, weight)
    torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    check_clipping(actual, (x, y, weight))


def test_order_selection_and_switches(monkeypatch):
    layer = mixed(64, False, torch.float64)
    assert [kernel.channel_order(group) for group in layer.groups].count("left") == 2
    for group in layer.groups:
        if group.connection_mode != "uvw":
            assert kernel.channel_order(group) is None
    small = mixed(4, False, torch.float64)
    assert all(kernel.channel_order(group) is None for group in small.groups)

    def reject(group):
        raise AssertionError("Contraction order should not be selected")

    monkeypatch.setattr(kernel, "channel_order", reject)
    values = inputs(small, 3, False, torch.float64)
    small(*values)
    monkeypatch.setattr(config, "TORCH_CARTESIAN_REORDER", False)
    make_fx(small)(*values)
    monkeypatch.setattr(config, "TORCH_CARTESIAN_REORDER", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    make_fx(small)(*values)


@pytest.mark.parametrize("per_edge", [False, True])
def test_dynamic_force_loss(per_edge, monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    monkeypatch.setattr(config, "TORCH_CARTESIAN_REORDER", True)
    monkeypatch.setattr(config, "CARTESIAN_REORDER_MIN_WORK", 64**3)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    layer = co3.ictd.TensorProduct(
        "64x1o",
        "64x2e",
        "64x3o",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=not per_edge,
    ).double()
    x, y, weight = (
        value.detach().contiguous().requires_grad_()
        for value in inputs(layer, 7, per_edge, torch.float64)
    )

    def joint(x, y, w):
        return force_loss(layer, x, y, w)

    count = ShapeVar("nodes", min=0, optimization_hint=7)
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec([count, None]),
                    "y": TensorSpec([count, None]),
                    "w": TensorSpec([count, None] if per_edge else [None]),
                }
            )
        ),
    )(x, y, weight)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = []

    def backend(module, values):
        from torch._inductor.compile_fx import compile_fx

        compiled.append(module)
        return compile_fx(module, values)

    run = torch.compile(graph, backend=backend, dynamic=True, fullgraph=True)
    for nodes in (7, 11, 9, 0):
        if nodes == 0:
            assert len(compiled) == 1
        x, y, fresh_weight = (
            value.detach().contiguous().requires_grad_()
            for value in inputs(layer, nodes, per_edge, torch.float64)
        )
        if per_edge:
            weight = fresh_weight
        for _ in range(2):
            with monkeypatch.context() as flags:
                flags.setattr(config, "TORCH_REFERENCE", True)
                expected = joint(x, y, weight)
            arguments = tuple(value.detach() for value in (x, y, weight))
            actual = run(*arguments)
            torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)
            check_clipping(actual, arguments)
            weight = (weight - 0.001 * expected[-1]).detach().requires_grad_()
