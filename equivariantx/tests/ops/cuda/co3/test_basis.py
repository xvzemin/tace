"""Native projection derivatives and standalone joint AOTI execution."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3
from equivariantx.ops import config
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 11])
def test_native_projection(dtype, rows):
    from equivariantx.ops.cuda.co3.basis import project_rank_two

    value = torch.randn(rows, 3, 9, 2, device="cuda", dtype=dtype)[..., 0]
    matrix = co3.Projector.projection_matrix(2).to(value)
    torch.testing.assert_close(project_rank_two(value), value @ matrix)


def test_high_order_and_transforms():
    layer = co3.Projector("2x2e+1x3e", backend="cuda").double().cuda()
    reference = co3.Projector("2x2e+1x3e").double().cuda()
    value = torch.randn(
        4, layer.irreps_in.dim, dtype=torch.float64, device="cuda", requires_grad=True
    )
    losses = [function(value).sin().sum() for function in (layer, reference)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, value, create_graph=True)[0] for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-12, rtol=3e-12)
        tangent = torch.randn_like(value) * 0.2
        losses = [(gradient * tangent).sum() for gradient in gradients]
    tangent = torch.randn_like(value)
    torch.testing.assert_close(
        torch.func.jvp(layer, (value,), (tangent,)),
        torch.func.jvp(reference, (value,), (tangent,)),
    )
    torch.testing.assert_close(
        torch.func.jvp(
            torch.func.grad(lambda x: layer(x).sin().sum()), (value,), (tangent,)
        ),
        torch.func.jvp(
            torch.func.grad(lambda x: reference(x).sin().sum()), (value,), (tangent,)
        ),
    )
    torch.testing.assert_close(torch.vmap(layer)(value), reference(value))
    from equivariantx.tests.utils.testing import check_forward_ad

    check_forward_ad(layer, reference, (value,))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_joint_aoti(tmp_path, backend):
    layer = co3.Projector("2x2e", backend=backend).double().cuda()
    value = torch.randn(5, 18, dtype=torch.float64, device="cuda", requires_grad=True)
    weight = torch.randn(18, dtype=torch.float64, device="cuda", requires_grad=True)

    def joint(value, weight):
        energy = layer(value * weight).sin().sum()
        force = -torch.autograd.grad(energy, value, create_graph=True)[0]
        loss = (force - 0.2).square().mean() + energy.square() * 0.01
        return loss, force, torch.autograd.grad(loss, weight)[0]

    graph = make_fx(joint)(value, weight)
    inputs = value.detach(), weight.detach()
    exported = torch.export.export(graph, inputs, strict=True)
    package = tmp_path / "projection.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = joint(value, weight)
        records.append(
            {
                "inputs": (value.detach().cpu(), weight.detach().cpu()),
                "output": tuple(x.detach().cpu() for x in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
        value = (value.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = runner(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(x.cpu() for x in actual), record['output'], atol=2e-10, rtol=2e-10)
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


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 7])
def test_change_basis_native(inverse, dtype, rows):
    layer = co3.ChangeOfBasis("2x0e+2x1o+3x2e+1x3e", inverse, backend="cuda")
    layer = layer.to(device="cuda", dtype=dtype)
    reference = co3.ChangeOfBasis(
        layer.irreps_in if inverse else layer.irreps_out, inverse, backend="torch"
    )
    reference = reference.to(device="cuda", dtype=dtype)
    value = torch.randn(rows, layer.input_dim, 2, device="cuda", dtype=dtype)[..., 0]
    precision = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.testing.assert_close(layer(value), reference(value))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = precision


@pytest.mark.parametrize("inverse", [False, True])
def test_change_basis_derivatives(inverse, monkeypatch):
    layer = co3.ChangeOfBasis("2x2e+1x2o+1x3e", inverse, backend="cuda").double().cuda()
    reference = (
        co3.ChangeOfBasis("2x2e+1x2o+1x3e", inverse, backend="torch").double().cuda()
    )
    value = (
        torch.randn(3, layer.input_dim, dtype=torch.float64, device="cuda") * 0.2
    ).requires_grad_()
    actual, expected = layer(value), reference(value)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        tangent = torch.randn_like(value) * 0.2
        actual, expected = [
            (
                torch.autograd.grad(y.sin().sum(), value, create_graph=True)[0]
                * tangent
            ).sum()
            for y in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    from equivariantx.tests.utils.testing import check_forward_ad

    check_forward_ad(layer, reference, (value,))
    torch.testing.assert_close(torch.vmap(layer)(value), reference(value))
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    graph = make_fx(layer)(value)
    ordinary = make_fx(reference)(value)
    assert any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    assert not any(str(node.target).startswith("eqx.") for node in ordinary.graph.nodes)
    torch.testing.assert_close(graph(value), reference(value), atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(
        ordinary(value), reference(value), atol=3e-12, rtol=3e-12
    )
    assert layer.backend == "cuda" and reference.backend == "torch"


def test_change_basis_dynamic_aoti(tmp_path):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    embed = co3.ChangeOfBasis("2x2e+1x3e", backend="cuda").double().cuda()
    extract = co3.ChangeOfBasis("2x2e+1x3e", True, backend="cuda").double().cuda()
    reference_embed = co3.ChangeOfBasis("2x2e+1x3e").double().cuda()
    reference_extract = co3.ChangeOfBasis("2x2e+1x3e", True).double().cuda()

    def operation(first, second):
        def joint(x, w):
            value = second(first(x * w).sin())
            energy = value.square().sum() * 0.01
            force = -torch.autograd.grad(energy, x, create_graph=True)[0]
            loss = (force - 0.1).square().mean() + energy.square()
            return loss, force, torch.autograd.grad(loss, w)[0]

        return joint

    joint = operation(embed, extract)
    reference = operation(reference_embed, reference_extract)
    x = (
        torch.randn(7, embed.input_dim, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    weight = (
        torch.randn(embed.input_dim, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
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
    assert any(
        str(node.target).startswith("eqx.grid_linear") for node in graph.graph.nodes
    )
    exported = torch.export.export(
        graph,
        (x.detach(), weight.detach()),
        dynamic_shapes=({0: torch.export.Dim("nodes", min=2)}, None),
        strict=True,
    )
    package = tmp_path / "basis.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for nodes in (7, 11, 9):
        x = (
            torch.randn(nodes, embed.input_dim, device="cuda", dtype=torch.float64)
            * 0.2
        ).requires_grad_()
        for _ in range(3):
            expected = reference(x, weight)
            torch.testing.assert_close(
                joint(x, weight), expected, atol=2e-10, rtol=2e-10
            )
            records.append(
                {
                    "inputs": (x.detach().cpu(), weight.detach().cpu()),
                    "output": tuple(t.detach().cpu() for t in expected),
                }
            )
            weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "records.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    actual = run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(x.cpu() for x in actual), record['output'], atol=2e-10, rtol=2e-10)
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
