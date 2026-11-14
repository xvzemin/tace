"""Native shared-channel joint AOTI packages and parameter updates."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.cuda.channel_linear import evaluate
from equivariantx.ops.common.torch.channel_linear import primal
from equivariantx.tests.ops.common.torch.test_channel_linear import linear_case
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native channel maps require CUDA"
)


@pytest.mark.parametrize("space", ["o2", "co2", "co3"])
def test_unbacked_empty_force_graph(space):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    layer, inputs = linear_case(space, "cuda", "cuda")
    metadata = layer.channel_metadata

    def loss_of(y, x):
        energy = y.sin().square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        return force, energy.square() * 0.01 + (force - 0.2).square().sum()

    def operation(x, weight, bias):
        y = evaluate(metadata, x, weight, bias)
        force, loss = loss_of(y, x)
        return y, force, loss, *torch.autograd.grad(loss, (weight, bias))

    rows = ShapeVar("rows", min=0, max=16)
    graph = make_fx(
        operation,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec({"x": TensorSpec((rows, None))})),
    )(*inputs)
    for count in (0, 1, 7, 11):
        shapes = ((count, inputs[0].shape[-1]), inputs[1].shape, inputs[2].shape)
        values = tuple(
            (
                torch.randn(
                    *shape[:-1], 2 * shape[-1], device="cuda", dtype=torch.float64
                )
                * 0.2
            )[..., ::2].requires_grad_()
            for shape in shapes
        )
        expected = primal(metadata, *values, ordinary=True)
        force, loss = loss_of(expected, values[0])
        expected_joint = expected, force, loss, *torch.autograd.grad(loss, values[1:])
        with torch.no_grad():
            actual = graph(*values)
        torch.testing.assert_close(actual, expected_joint, atol=3e-10, rtol=3e-9)
        outputs = [
            evaluate(metadata, *values),
            primal(metadata, *values, ordinary=True),
        ]
        losses = [loss_of(y, values[0])[1] for y in outputs]
        directions = tuple(torch.randn_like(v) * 0.1 for v in values)
        for _ in range(5):
            gradients = [
                torch.autograd.grad(loss, values, create_graph=True) for loss in losses
            ]
            torch.testing.assert_close(*gradients, atol=3e-10, rtol=3e-9)
            losses = [
                sum((g * d).sum() for g, d in zip(gradient, directions))
                for gradient in gradients
            ]


@pytest.mark.parametrize("space", ["o2", "co2", "co3"])
def test_joint_aoti(tmp_path, space):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    layer, inputs = linear_case(space, "cuda", "cuda")
    metadata = layer.channel_metadata

    def operation(x, w, b, ordinary=False):
        y = (
            primal(metadata, x, w, b, ordinary=True)
            if ordinary
            else evaluate(metadata, x, w, b)
        )
        energy = y.sin().square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = (force - 0.2).square().sum() + energy.square() * 0.01
        return y, force, loss, *torch.autograd.grad(loss, (w, b))

    def captured(x, weight, bias):
        return operation(x, weight, bias)

    rows = ShapeVar("rows", min=0, max=16)
    graph = make_fx(
        captured,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec({"x": TensorSpec((rows, None))})),
    )(*inputs)
    for count in (0, 1):
        x = (
            torch.randn(count, inputs[0].shape[-1], device="cuda", dtype=torch.float64)
            * 0.2
        ).requires_grad_()
        expected = operation(x, *inputs[1:], ordinary=True)
        with torch.no_grad():
            actual = graph(x, *inputs[1:])
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-9)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in inputs),
        strict=True,
        dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=16)},)
        + (None,) * (len(inputs) - 1),
    )
    package = tmp_path / "linear.pt2"
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(package),
        inductor_configs={
            "aot_inductor.check_lowerbound": True,
            "aot_inductor.check_upperbound": True,
        },
    )
    parameters = list(inputs[1:])
    initial = tuple(value.detach().cpu().clone() for value in parameters)
    records = []
    for count in (7, 11, 5):
        x = (
            torch.randn(count, inputs[0].shape[-1], device="cuda", dtype=torch.float64)
            * 0.2
        ).requires_grad_()
        expected = operation(x, *parameters, ordinary=True)
        records.append(
            {
                "x": x.detach().cpu(),
                "output": tuple(value.detach().cpu().clone() for value in expected),
            }
        )
        for value, gradient in zip(parameters, expected[3:]):
            value.grad = gradient
        assert torch.nn.utils.clip_grad_norm_(parameters, 0.001, foreach=True) > 0.001
        with torch.no_grad():
            for value in parameters:
                value.add_(value.grad, alpha=-0.01)
    final = tuple(value.detach().cpu() for value in parameters)
    assert all(not torch.equal(a, b) for a, b in zip(initial, final))
    data = tmp_path / "updates.pt"
    torch.save({"initial": initial, "steps": records, "final": final}, data)
    script = """
import sys
import torch
torch.set_num_threads(2)
import equivariantx
run = torch._inductor.aoti_load_package(sys.argv[1])
record = torch.load(sys.argv[2], weights_only=True)
parameters = [value.cuda() for value in record['initial']]
for step in record['steps']:
    actual = run(step['x'].cuda(), *parameters)
    torch.testing.assert_close(tuple(value.cpu() for value in actual), step['output'], atol=3e-10, rtol=3e-9)
    gradients = actual[3:]
    intervals = [(g.data_ptr(), g.data_ptr() + g.numel()*g.element_size()) for g in gradients]
    for i, a in enumerate(intervals):
        for b in intervals[i+1:]:
            assert a[1] <= b[0] or b[1] <= a[0]
    for value, gradient in zip(parameters, gradients):
        assert value.untyped_storage().data_ptr() != gradient.untyped_storage().data_ptr()
        value.grad = gradient
    assert torch.nn.utils.clip_grad_norm_(parameters, 0.001, foreach=True) > 0.001
    for value in parameters:
        value.add_(value.grad, alpha=-0.01)
torch.testing.assert_close(tuple(value.cpu() for value in parameters), record['final'], atol=3e-11, rtol=3e-9)
for count in (0, 1, 17):
    try:
        run(step['x'].new_zeros(count, step['x'].shape[-1]).cuda(), *parameters)
    except RuntimeError as error:
        assert ('too small' if count < 2 else 'too large') in str(error)
        assert ('>= 2' if count < 2 else '<= 16') in str(error)
    else:
        raise AssertionError('AOTI input bounds must remain enforced.')
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "AOTI_RUNTIME_CHECK_INPUTS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("tf32", [False, True])
def test_native_fp32_policy(tf32):
    from equivariantx import co3

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        layer = co3.Linear("96x1o", "128x1o", backend="cuda").cuda()
        x = (torch.randn(7, 288, device="cuda") * 0.1).requires_grad_()
        w = (torch.randn(layer.weight_numel, device="cuda") * 0.1).requires_grad_()
        b = torch.empty(0, device="cuda")
        actual = layer(x, w, b)
        expected = primal(
            layer.channel_metadata, x.double(), w.double(), b.double(), ordinary=True
        )
        error = (actual.double() - expected).norm() / expected.norm()
        assert error < (1e-3 if tf32 else 1e-6)
        actual_gradient = torch.autograd.grad(actual.sin().sum(), (x, w))
        expected_gradient = torch.autograd.grad(expected.sin().sum(), (x, w))
        for a, e in zip(actual_gradient, expected_gradient):
            assert (a - e).norm() / e.norm() < (2e-3 if tf32 else 2e-6)
        with pytest.raises(TypeError):
            evaluate(layer.channel_metadata, x.half(), w.half(), b.half())
        with pytest.raises(ValueError):
            evaluate(layer.channel_metadata, x[:, :-1], w, b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
