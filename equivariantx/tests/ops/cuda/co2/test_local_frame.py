"""Native rectangular frame contractions in independent joint AOTI packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config
from equivariantx.ops.common.torch.tensor_axis import transform
from equivariantx.ops.compiler import compile_and_package
from equivariantx.ops.cuda.co2.frame_matrix import evaluate
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native Cartesian frames require CUDA"
)


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_cartesian_matrix_derivatives(rank):
    device = {"device": "cuda", "dtype": torch.float64}
    rows = torch.randn(10, 2 * 3**rank, **device)[:, ::2].requires_grad_()
    r = (torch.randn(2, 3, 3, **device) * 0.3).transpose(-1, -2).requires_grad_()
    values = rows, r
    actual = evaluate(rows, r, rank)
    expected = transform(rows, r.transpose(-1, -2).unsqueeze(-3), rank)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    losses = [v.sin().sum() for v in (actual, expected)]
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-9, rtol=3e-10)
        tangents = [torch.randn_like(v) * 0.1 for v in values]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


@pytest.mark.parametrize(
    "wide,precision", [(False, "fp64"), (True, "fp64"), (True, "fp32"), (True, "tf32")]
)
def test_joint_aoti(monkeypatch, request, tmp_path, wide, precision):
    monkeypatch.setattr(config, "TENSOR_AXIS_INLINE", wide)
    previous = torch.backends.cuda.matmul.allow_tf32
    request.addfinalizer(
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", previous)
    )
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    dtype = torch.float64 if precision == "fp64" else torch.float32
    irreps = "64x2o+64x3o" if wide else "2x1o+1x2e"
    layers = {
        backend: co2.LocalFrame(irreps, mmax=2, backend=backend).to(
            device="cuda", dtype=torch.float64 if backend == "torch" else dtype
        )
        for backend in ("torch", "cuda")
    }

    def operation(x, rotation, weight, ordinary=False):
        layer = layers["torch" if ordinary else "cuda"]
        y = layer(x * weight, rotation)
        if wide:
            y = layer.to_global(y.sin(), rotation)
        energy = y.sin().sum() * 0.1
        force = -torch.autograd.grad(energy, rotation, create_graph=True)[0]
        loss = (force - 0.01).square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    width = layers["cuda"].input_dim
    x = (torch.randn(3, width, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    rotation = (torch.randn(3, 3, 3, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    weight = (torch.randn(width, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    values = x, rotation, weight

    def joint(x, rotation, weight):
        return operation(x, rotation, weight)

    dynamic_shapes = None
    if wide:
        pytest.importorskip("torch.fx.experimental.dynamic_spec")
        from torch.fx.experimental.dynamic_spec import (
            ParamsSpec,
            ShapesSpec,
            ShapeVar,
            TensorSpec,
        )

        nodes = ShapeVar("frames", min=2, optimization_hint=3)
        shapes = ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec([nodes, None]),
                    "rotation": TensorSpec([nodes, None, None]),
                    "weight": TensorSpec([None]),
                }
            )
        )
        graph = make_fx(
            joint,
            tracing_mode="fake",
            dynamic_shapes=shapes,
            _allow_non_fake_inputs=True,
        )(*values)
        count = torch.export.Dim("frames", min=2, max=16)
        dynamic_shapes = ({0: count}, {0: count}, None)
    else:
        graph = make_fx(joint)(*values)
    targets = {node.target for node in graph.graph.nodes}
    if wide:
        assert torch.ops.eqx.cartesian_frame_matrix.default in targets
        assert torch.ops.eqx.cartesian_frame_rows.default in targets
    else:
        assert torch.ops.eqx.tensor_axis.default in targets
        assert torch.ops.eqx.tensor_axis_matrix_gradient.default in targets
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph,
        tuple(v.detach() for v in values),
        strict=True,
        dynamic_shapes=dynamic_shapes,
    )
    package = tmp_path / "local-frame.pt2"
    compile_and_package(exported, package_path=str(package))
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for rows in (3, 7, 5) if wide else (3, 3, 3):
        x = (x.new_empty(rows, width).normal_() * 0.2).requires_grad_()
        rotation = (rotation.new_empty(rows, 3, 3).normal_() * 0.2).requires_grad_()
        original = x, rotation, weight
        expected = operation(
            *(v.detach().double().requires_grad_() for v in original), True
        )
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in original),
                "output": tuple(v.detach().cpu() for v in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1].to(weight)).detach().requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
precision=sys.argv[3]
torch.backends.cuda.matmul.allow_tf32=precision=="tf32"
tolerance={"fp64":3e-9,"fp32":5e-5,"tf32":5e-3}[precision]
run=torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad(): actual=run(*(value.cuda() for value in record['inputs']))
    for value,expected in zip(actual,record['output'],strict=True):
        torch.testing.assert_close(value.double().cpu(),expected,atol=tolerance,rtol=tolerance)
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script(script),
            str(package),
            str(data),
            precision,
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
