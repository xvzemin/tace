"""Channelwise path algebra, broadcasts and recursive functional derivatives."""

import pytest
import torch

from equivariantx import co2, o2
from equivariantx.ops import config
from equivariantx.ops.torch.co2.linear import uu_reference as cartesian_reference
from equivariantx.ops.torch.o2.linear import uu_reference as circular_reference


def linear_case(space, mode, backend, device, weights="node"):
    layer = {"o2": o2, "co2": co2}[space].UuLinear(
        "4x0e+2x0e+4x1m+2x2m",
        "4x0e+2x1m+2x2m+2x0e" + ("+2x0o" if mode == "sum" else ""),
        2,
        path_mode=mode,
        backend=backend,
    )
    x = torch.randn(3, layer.irreps_in.dim, device=device, dtype=torch.float64) * 0.2
    shape = {"shared": (), "single": (1,), "node": (3,)}[weights]
    w = (
        torch.randn(*shape, layer.weight_numel, device=device, dtype=torch.float64)
        * 0.2
    )
    reference = circular_reference if space == "o2" else cartesian_reference
    return (
        layer,
        (x.requires_grad_(), w.requires_grad_()),
        lambda x, w: reference(layer, x, w),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("mode", ["sum", "expand"])
@pytest.mark.parametrize("weights", ["shared", "single", "node", "leading"])
def test_unbacked_broadcast_adjoint(device, space, mode, weights):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops.common.torch.uu_linear import evaluate

    layer, _, reference = linear_case(space, mode, "torch", device)

    def data(count):
        prefix = {"shared": (), "single": (1,), "node": (count,), "leading": (1, 3, 1)}[
            weights
        ]
        shapes = ((2, 1, count, layer.irreps_in.dim), (*prefix, layer.weight_numel))
        return tuple(
            (
                torch.randn(
                    *shape[:-1], shape[-1] * 2, device=device, dtype=torch.float64
                )
                * 0.2
            )[..., ::2].requires_grad_()
            for shape in shapes
        )

    def loss_of(y, x):
        energy = y.sin().square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        return force, energy.square() * 0.01 + (force - 0.2).square().sum()

    def operation(x, weight):
        y = evaluate(layer.uu_metadata, x, weight)
        force, loss = loss_of(y, x)
        return y, force, loss, torch.autograd.grad(loss, weight)[0]

    rows = ShapeVar("rows", min=0, max=16)
    specs = {"x": TensorSpec((None, None, rows, None))}
    if weights == "node":
        specs["weight"] = TensorSpec((rows, None))
    graph = make_fx(
        operation,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec(specs)),
    )(*data(3))
    for count in (0, 1, 7):
        inputs = data(count)
        expected = reference(*inputs)
        force, loss = loss_of(expected, inputs[0])
        expected_joint = expected, force, loss, torch.autograd.grad(loss, inputs[1])[0]
        with torch.no_grad():
            actual_joint = graph(*inputs)
        torch.testing.assert_close(actual_joint, expected_joint, atol=3e-10, rtol=3e-9)
        outputs = [evaluate(layer.uu_metadata, *inputs), reference(*inputs)]
        assert type(outputs[0].grad_fn).__name__ == "UuLinearBackward"
        losses = [loss_of(y, inputs[0])[1] for y in outputs]
        directions = tuple(torch.randn_like(value) * 0.1 for value in inputs)
        for _ in range(5):
            gradients = [
                torch.autograd.grad(value, inputs, create_graph=True)
                for value in losses
            ]
            torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-9)
            losses = [
                sum((g * d).sum() for g, d in zip(gradient, directions))
                for gradient in gradients
            ]
        if count == 1:
            torch.testing.assert_close(
                torch.func.jvp(
                    lambda x, w: evaluate(layer.uu_metadata, x, w), inputs, directions
                ),
                torch.func.jvp(reference, inputs, directions),
                atol=3e-11,
                rtol=3e-11,
            )


@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("mode", ["sum", "expand"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("weights", ["shared", "single", "node"])
def test_recursive_paths(monkeypatch, space, mode, backend, weights, device):
    monkeypatch.setattr(config, "TORCH_UU_LINEAR", True)
    monkeypatch.setattr(config, "TORCH_UU_ADJOINT", True)
    layer, inputs, reference = linear_case(space, mode, backend, device, weights)
    actual, expected = layer(*inputs), reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    losses = [value.sin().sum() for value in (actual, expected)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
        tangents = tuple(torch.randn_like(value) * 0.1 for value in inputs)
        losses = [
            sum((g * t).sum() for g, t in zip(gradient, tangents))
            for gradient in gradients
        ]
    tangents = tuple(torch.randn_like(value) * 0.1 for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(layer, inputs, tangents),
        torch.func.jvp(reference, inputs, tangents),
        atol=3e-11,
        rtol=3e-11,
    )

    def differentiated(function):
        return torch.func.grad(lambda x, w: function(x, w).sin().sum(), argnums=(0, 1))

    torch.testing.assert_close(
        torch.func.jvp(differentiated(layer), inputs, tangents),
        torch.func.jvp(differentiated(reference), inputs, tangents),
        atol=3e-10,
        rtol=3e-10,
    )
    x, w = inputs
    if weights == "node":
        torch.testing.assert_close(
            torch.vmap(layer)(x, w), layer(x, w), atol=3e-11, rtol=3e-11
        )
    else:
        torch.testing.assert_close(
            torch.vmap(lambda v: layer(v, w.reshape(-1)))(x),
            layer(x, w),
            atol=3e-11,
            rtol=3e-11,
        )
    values = (x[:0], w[:0] if weights == "node" else w)
    assert layer(*values).shape == (0, layer.irreps_out.dim)


@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("mode", ["sum", "expand"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_dynamic_compile(monkeypatch, space, mode, backend, device):
    monkeypatch.setattr(config, "TORCH_UU_LINEAR", True)
    monkeypatch.setattr(config, "TORCH_UU_ADJOINT", True)
    layer, inputs, reference = linear_case(space, mode, backend, device)
    compiled = torch.compile(layer, backend="aot_eager", dynamic=True, fullgraph=True)
    for rows in (3, 7):
        values = tuple(
            torch.randn(
                rows,
                v.shape[-1],
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            )
            for v in inputs
        )
        actual, expected = compiled(*values), reference(*values)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        gradients = [
            torch.autograd.grad(y.sin().sum(), values) for y in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("mode", ["sum", "expand"])
def test_broadcast_legacy_and_empty(monkeypatch, space, mode, device):
    monkeypatch.setattr(config, "TORCH_UU_LINEAR", True)
    monkeypatch.setattr(config, "TORCH_UU_ADJOINT", True)
    layer, inputs, reference = linear_case(space, mode, "cuda", device)
    x, w = inputs
    values = x.unsqueeze(0), w.unsqueeze(1)
    actual, expected = layer(*values), reference(*values)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    gradients = [torch.autograd.grad(y.sin().sum(), inputs) for y in (actual, expected)]
    torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
    del layer.uu_metadata
    del layer.backend
    torch.testing.assert_close(layer(*inputs), reference(*inputs))
    for irreps_in, irreps_out in [("", "2x0e"), ("2x1m", "2x0e"), ("2x0e", "")]:
        if mode == "expand" and irreps_out:
            with pytest.raises(ValueError, match="multiplicities must be positive"):
                {"o2": o2, "co2": co2}[space].UuLinear(
                    irreps_in, irreps_out, 2, path_mode=mode, backend="cuda"
                )
            continue
        empty = {"o2": o2, "co2": co2}[space].UuLinear(
            irreps_in, irreps_out, 2, path_mode=mode, backend="cuda"
        )
        values = (
            torch.randn(
                3,
                empty.irreps_in.dim,
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.randn(
                empty.weight_numel,
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            ),
        )
        result = empty(*values)
        assert not result.count_nonzero()
        for grad in torch.autograd.grad(result.sum(), values):
            assert not grad.count_nonzero()


@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_inductor_and_noncontiguous(monkeypatch, space, backend, device):
    monkeypatch.setattr(config, "TORCH_UU_LINEAR", True)
    monkeypatch.setattr(config, "TORCH_UU_ADJOINT", True)
    layer, inputs, reference = linear_case(space, "sum", backend, device)
    values = tuple(
        torch.stack((v.detach(), v.detach()), -1)[..., 0].requires_grad_()
        for v in inputs
    )
    actual, expected = layer(*values), reference(*values)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sin().sum(), values),
        torch.autograd.grad(expected.sin().sum(), values),
        atol=3e-10,
        rtol=3e-10,
    )
    compiled = torch.compile(layer, backend="inductor", fullgraph=True, dynamic=True)
    for rows in (3, 7):
        data = tuple(
            torch.randn(
                rows,
                v.shape[-1],
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            )
            * 0.2
            for v in inputs
        )
        actual, expected = compiled(*data), reference(*data)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), data),
            torch.autograd.grad(expected.sin().sum(), data),
            atol=3e-10,
            rtol=3e-10,
        )


@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("mode", ["sum", "expand"])
def test_single_and_wide_copy_axes(monkeypatch, space, mode, device):
    monkeypatch.setattr(config, "TORCH_UU_LINEAR", True)
    monkeypatch.setattr(config, "TORCH_UU_ADJOINT", True)
    namespace = {"o2": o2, "co2": co2}[space]
    reference = circular_reference if space == "o2" else cartesian_reference
    for ni, no in ((1, 1), (5, 6)):
        layer = namespace.UuLinear(
            f"{2 * ni}x1m", f"{2 * no}x1m", 2, path_mode=mode, backend="cuda"
        )
        inputs = (
            torch.randn(
                3,
                layer.irreps_in.dim,
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            )
            * 0.2,
            torch.randn(
                3,
                layer.weight_numel,
                device=device,
                dtype=torch.float64,
                requires_grad=True,
            )
            * 0.2,
        )
        outputs = [layer(*inputs), reference(layer, *inputs)]
        torch.testing.assert_close(*outputs, atol=3e-11, rtol=3e-11)
        for _ in range(2):
            gradients = [
                torch.autograd.grad(value.sin().sum(), inputs, create_graph=True)
                for value in outputs
            ]
            torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-10)
            outputs = [
                sum(g.square().sum() for g in gradient) for gradient in gradients
            ]
