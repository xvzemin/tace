"""Native planar reductions and independently loaded force-loss packages."""

import os
import subprocess
import sys

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
from equivariantx.tests.utils.testing import check_forward_ad, package_script

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("rows", [0, 7])
def test_change_basis(inverse, rows):
    layer = (
        co2.ChangeOfBasis("1x0e+3x1mo+2x4m", inverse, backend="cuda").cuda().double()
    )
    reference = co2.ChangeOfBasis("1x0e+3x1mo+2x4m", inverse).cuda().double()
    x = (
        torch.randn(rows, layer.input_dim, 2, device="cuda", dtype=torch.float64)[
            ..., 0
        ]
        * 0.2
    ).requires_grad_()
    actual, expected = layer(x), reference(x)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        direction = torch.randn_like(x) * 0.1
        actual, expected = [
            torch.autograd.grad(y.sin().sum(), x, create_graph=True)[0] * direction
            for y in (actual, expected)
        ]
    if rows:
        check_forward_ad(layer, reference, (x,))
        torch.testing.assert_close(torch.vmap(layer)(x), reference(x))


@pytest.mark.parametrize("rank", [2, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 7])
def test_native_projection(rank, dtype, rows):
    from equivariantx.ops.cuda.co2.basis import planar_project

    x = torch.randn(rows, 3, 2**rank, 2, device="cuda", dtype=dtype)[..., 0]
    expected = x @ co2.projector_matrix(rank).to(x)
    torch.testing.assert_close(planar_project(x, rank), expected)


def test_native_transforms_and_high_order(monkeypatch):
    monkeypatch.setattr(config, "PLANAR_PROJECTION_NATIVE", True)
    monkeypatch.setattr(config, "PLANAR_PROJECTION_INLINE", False)
    layer = co2.Projector("2x4m+1x0e+1x6m", backend="cuda").cuda().double()
    reference = co2.Projector("2x4m+1x0e+1x6m").cuda().double()
    x = (
        torch.randn(3, layer.input_dim, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    values = layer(x), reference(x)
    for _ in range(5):
        torch.testing.assert_close(*values, atol=2e-11, rtol=2e-11)
        direction = torch.randn_like(x) * 0.1
        values = [
            torch.autograd.grad(v.sin().sum(), x, create_graph=True)[0] * direction
            for v in values
        ]
    torch.testing.assert_close(*values, atol=2e-11, rtol=2e-11)
    check_forward_ad(layer, reference, (x,))
    torch.testing.assert_close(torch.vmap(layer)(x), reference(x))


@pytest.mark.parametrize("kind", ["inline", "native", "basis"])
def test_dynamic_joint_aoti(kind, monkeypatch, tmp_path):
    native = kind == "native"
    monkeypatch.setattr(config, "PLANAR_PROJECTION_INLINE", not native)
    monkeypatch.setattr(config, "PLANAR_PROJECTION_NATIVE", native)
    if kind == "basis":
        embedding = co2.ChangeOfBasis("2x4m", backend="cuda")
        layer = (
            torch.nn.Sequential(
                embedding,
                torch.nn.SiLU(),
                co2.ChangeOfBasis("2x4m", inverse=True, backend="cuda"),
            )
            .cuda()
            .double()
        )
        input_dim = embedding.input_dim
    else:
        layer = co2.Projector("2x4m", backend="cuda").cuda().double()
        input_dim = layer.input_dim
    x = (
        torch.randn(7, input_dim, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    weight = (
        torch.randn(input_dim, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()

    def joint(x, w):
        energy = layer(x * w).sin().sum() * 0.1
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, torch.autograd.grad(loss, w)[0]

    count = ShapeVar("nodes", min=2, optimization_hint=7)
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec({"x": TensorSpec([count, None]), "w": TensorSpec([None])})
        ),
    )(x, weight)
    assert (
        any("eqx.planar_project" in str(n.target) for n in graph.graph.nodes) == native
    )
    if kind == "basis":
        assert any("eqx.grid_linear" in str(n.target) for n in graph.graph.nodes)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    inputs = x.detach(), weight.detach()
    exported = torch.export.export(
        graph,
        inputs,
        dynamic_shapes=({0: torch.export.Dim("nodes", min=2)}, {}),
        strict=True,
    )
    package = tmp_path / "planar.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for rows in (7, 11, 9):
        x = (
            torch.randn(rows, input_dim, device="cuda", dtype=torch.float64) * 0.2
        ).requires_grad_()
        for _ in range(3):
            with monkeypatch.context() as flags:
                flags.setattr(config, "TORCH_REFERENCE", True)
                expected = joint(x, weight)
            records.append(
                {
                    "inputs": (x.detach().cpu(), weight.detach().cpu()),
                    "output": tuple(v.detach().cpu() for v in expected),
                }
            )
            weight = (weight - expected[-1] * 0.01).detach().requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    actual = run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), record['output'], atol=2e-10, rtol=2e-10)
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
