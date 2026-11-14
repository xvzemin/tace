"""Native channelwise AOTI packages independent of EQX at runtime."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.cuda.uu_linear import evaluate
from equivariantx.tests.ops.common.torch.test_uu_linear import linear_case

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native UuLinear requires CUDA"
)


@pytest.mark.parametrize("space", ["o2", "co2"])
@pytest.mark.parametrize("mode", ["sum", "expand"])
@pytest.mark.parametrize("weights", ["shared", "single", "node"])
def test_unbacked_empty_force_graph(space, mode, weights):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    layer, inputs, reference = linear_case(space, mode, "cuda", "cuda", weights)

    def loss_of(y, x):
        energy = y.sin().square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        return force, energy.square() * 0.01 + (force - 0.2).square().sum()

    def operation(x, weight):
        y = evaluate(layer.uu_metadata, x, weight)
        force, loss = loss_of(y, x)
        return y, force, loss, torch.autograd.grad(loss, weight)[0]

    rows = ShapeVar("rows", min=0, max=16)
    specs = {"x": TensorSpec((rows, None))}
    if weights == "node":
        specs["weight"] = TensorSpec((rows, None))
    graph = make_fx(
        operation,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec(specs)),
    )(*inputs)
    for count in (0, 1, 7, 11):
        prefix = {"shared": (), "single": (1,), "node": (count,)}[weights]
        shapes = ((count, inputs[0].shape[-1]), (*prefix, layer.weight_numel))
        values = tuple(
            (
                torch.randn(
                    *shape[:-1], 2 * shape[-1], device="cuda", dtype=torch.float64
                )
                * 0.2
            )[..., ::2].requires_grad_()
            for shape in shapes
        )
        expected = reference(*values)
        force, loss = loss_of(expected, values[0])
        expected_joint = expected, force, loss, torch.autograd.grad(loss, values[1])[0]
        with torch.no_grad():
            actual = graph(*values)
        torch.testing.assert_close(actual, expected_joint, atol=3e-10, rtol=3e-9)
        losses = [
            loss_of(y, values[0])[1]
            for y in (evaluate(layer.uu_metadata, *values), reference(*values))
        ]
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


@pytest.mark.parametrize(
    "space,mode,weights",
    [
        ("o2", "sum", "node"),
        ("o2", "expand", "node"),
        ("co2", "sum", "node"),
        ("co2", "expand", "node"),
        ("o2", "sum", "single"),
        ("co2", "sum", "shared"),
    ],
)
def test_joint_aoti_without_eqx(tmp_path, space, mode, weights):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    layer, inputs, reference = linear_case(space, mode, "cuda", "cuda", weights)

    def operation(x, weight, ordinary=False):
        w = (
            weight[torch.arange(x.shape[0], device=x.device) % 3]
            if weights == "node"
            else weight
        )
        y = reference(x, w) if ordinary else evaluate(layer.uu_metadata, x, w)
        energy = y.sin().square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = (force - 0.2).square().sum() + energy.square() * 0.01
        return y, force, loss, torch.autograd.grad(loss, weight)[0]

    def captured(x, weight):
        return operation(x, weight)

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
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(package), str(data)],
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


def test_native_fp32_and_validation():
    for space in ("o2", "co2"):
        layer, inputs, reference = linear_case(space, "sum", "cuda", "cuda")
        actual = layer(*(v.detach().float() for v in inputs))
        expected = reference(*inputs)
        torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-5)
        with pytest.raises(TypeError):
            evaluate(layer.uu_metadata, *(v.half() for v in inputs))
        with pytest.raises(ValueError):
            evaluate(layer.uu_metadata, inputs[0][:, :-1], inputs[1])
