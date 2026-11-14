"""Sparse spherical operators in complete joint parameter gradients."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config
from equivariantx.ops.torch.co2.spherical import reference


@pytest.mark.parametrize("method", ["chebyshev", "recurrence"])
def test_joint_inductor(method, monkeypatch, device):
    monkeypatch.setattr(config, "TORCH_SPHERICAL_COUPLING", True)
    monkeypatch.setattr(config, "TORCH_SPHERICAL_ADJOINT", True)
    layer = co2.SphericalCoupling(2, 3, 2, method=method).to(
        device=device, dtype=torch.float64
    )
    x = (
        torch.randn(2, 3, 5, device=device, dtype=torch.float64) * 0.1
    ).requires_grad_()
    v = (
        torch.randn(2, 1, 3, device=device, dtype=torch.float64) + 0.5
    ).requires_grad_()
    w = (torch.randn(3, 1, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    def run(x, v, w, ordinary=False):
        y = reference(layer, x * w, v, method) if ordinary else layer(x * w, v)
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(lambda *args: run(*args))(x, v, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    for _ in range(3):
        actual = compiled(x.detach(), v.detach(), w.detach())
        expected = run(x, v, w, True)
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()


def test_shared_matrix_static_selection():
    from equivariantx.ops.torch.co2.spherical import shared_matrix

    assert shared_matrix(torch.empty(64, 5), torch.empty(3))
    assert shared_matrix(torch.empty(7, 64, 5), torch.empty(7, 1, 3))
    assert not shared_matrix(torch.empty(7, 5), torch.empty(3))
    assert not shared_matrix(torch.empty(64, 5), torch.empty(64, 3))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("method", ["chebyshev", "recurrence"])
@pytest.mark.parametrize("shared_direction", [False, True])
def test_unbacked_joint_package(
    backend, method, shared_direction, device, tmp_path, monkeypatch
):
    """Check empty FX rows separately from the package's bounded inputs."""
    if backend == "cuda" and device != "cuda":
        pytest.skip("Native kernels require CUDA")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    monkeypatch.setattr(config, "SPHERICAL_INLINE", backend != "cuda")
    layer = co2.SphericalCoupling(2, 3, 2, method=method, backend=backend).to(
        device=device, dtype=torch.float64
    )

    def inputs(count):
        x = (torch.randn(count, 10, device=device, dtype=torch.float64) * 0.1)[
            :, ::2
        ].requires_grad_()
        shape = (6,) if shared_direction else (count, 6)
        v = (torch.randn(shape, device=device, dtype=torch.float64) * 0.1 + 0.5)[
            ..., ::2
        ].requires_grad_()
        return x, v

    weight = (torch.randn(5, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    def joint(x, v, w, ordinary=False):
        y = reference(layer, x * w, v, method) if ordinary else layer(x * w, v)
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, torch.autograd.grad(loss, w)[0]

    values = *inputs(7), weight
    rows = ShapeVar("rows", min=0, optimization_hint=7)
    specs = {
        "x": TensorSpec([rows, None]),
        "v": TensorSpec([None]) if shared_direction else TensorSpec([rows, None]),
        "w": TensorSpec([None]),
    }
    graph = make_fx(
        lambda x, v, w: joint(x, v, w),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec(specs)),
    )(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    rows = torch.export.Dim("rows", min=2, max=64)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in values),
        dynamic_shapes=({0: rows}, {} if shared_direction else {0: rows}, {}),
        strict=True,
    )
    package = tmp_path / "spherical-dynamic.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    initial = weight.detach().cpu().clone()
    records = []
    for count in (7, 11, 33, 0, 1):
        values = *inputs(count), weight
        for _ in range(3):
            expected = joint(*values, ordinary=True)
            torch.testing.assert_close(graph(*values), expected, atol=3e-9, rtol=3e-9)
            if count < 2:
                continue
            records.append(
                dict(
                    inputs=tuple(value.detach().cpu().clone() for value in values[:2]),
                    strides=tuple(value.stride() for value in values[:2]),
                    outputs=tuple(value.detach().cpu().clone() for value in expected),
                )
            )
            with torch.no_grad():
                weight.add_(expected[-1], alpha=-0.01)
    data = tmp_path / "records.pt"
    torch.save((initial, records), data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
weight, records = torch.load(sys.argv[2], weights_only=True)
weight = weight.to(sys.argv[3])
for record in records:
    # Preserve the captured strided layouts when copying CPU records to CUDA.
    values = []
    for value, stride in zip(record['inputs'], record['strides']):
        target = torch.empty_strided(value.shape, stride, dtype=value.dtype, device=sys.argv[3])
        target.copy_(value)
        values.append(target)
    saved = tuple(value.clone() for value in (*values, weight))
    actual = run(*values, weight)
    torch.testing.assert_close(tuple(value.cpu() for value in actual), record['outputs'], atol=3e-9, rtol=3e-9)
    gradient = actual[-1].clone()
    outputs = tuple(value.clone() for value in actual[:-1])
    torch._foreach_mul_([actual[-1]], 0.125)
    torch.testing.assert_close((*values, weight), saved, atol=0, rtol=0)
    torch.testing.assert_close(actual[:-1], outputs, atol=0, rtol=0)
    weight.add_(gradient, alpha=-0.01)
    assert torch.count_nonzero(actual[1]) > 0
    assert not torch.equal(weight, saved[-1])
for count in (0, 1, 65):
    invalid = []
    for value in values:
        shape = (count, *value.shape[1:]) if value.ndim == 2 else value.shape
        invalid.append(torch.empty_strided(shape, value.stride(), dtype=value.dtype, device=value.device))
    try:
        run(*invalid, weight)
    except RuntimeError as error:
        assert ('too small' if count < 2 else 'too large') in str(error)
    else:
        raise AssertionError('The exported row bounds were not enforced')
assert 'equivariantx' not in sys.modules and 'e3nn' not in sys.modules
""",
            str(package),
            str(data),
            device,
        ],
        cwd=tmp_path,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "AOTI_RUNTIME_CHECK_INPUTS": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
