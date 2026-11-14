"""Native factorized kernels and independent AOTI force-training packages."""

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

from equivariantx.ops import config
from equivariantx.tests.ops.torch.o3.test_tensor_product import (
    convolution_training_problem,
    training_problem,
)
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@pytest.mark.parametrize("kind", ["gaunt", "matrix"])
def test_native_joint_package(kind, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "FACTORIZED_INLINE", False)
    operation, values = training_problem(kind, "cuda", "cuda")
    operation(*values)
    graph = make_fx(operation)(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(v.detach() for v in values), strict=True
    )
    package = tmp_path / "factorized.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        expected = operation(*values)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in values),
                "outputs": tuple(v.detach().cpu() for v in expected),
            }
        )
        values = (
            *values[:2],
            (values[2] - 0.01 * expected[-1]).detach().requires_grad_(),
        )
    data = tmp_path / "values.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual), row['outputs'], atol=3e-9, rtol=3e-9)
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["last", "full"])
@pytest.mark.parametrize("width", [64, 128, 1024])
def test_radial_joint_package(mode, width, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "EDGE_PROGRAM_INLINE", True)
    dynamic = mode == "full"
    operation, values = convolution_training_problem(mode, width, dynamic=dynamic)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    operation(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    export_shapes = None
    if dynamic:

        def joint(x, y, radial, edges, *parameters):
            return operation(x, y, radial, edges, *parameters)

        edges = ShapeVar("edges", min=2)
        shapes = ShapesSpec(
            ParamsSpec(
                {
                    "y": TensorSpec((edges, None)),
                    "radial": TensorSpec((edges, None)),
                    "edges": TensorSpec((None, edges)),
                }
            )
        )
        graph = make_fx(
            joint,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
            dynamic_shapes=shapes,
        )(*values)
        edges = torch.export.Dim("edges", min=2)
        export_shapes = (None, {0: edges}, {0: edges}, {1: edges}) + (None,) * 4
    else:
        graph = make_fx(operation)(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert "eqx.edge_program" not in graph.code
    exported = torch.export.export(
        graph,
        tuple(v.detach() for v in values),
        strict=True,
        dynamic_shapes=export_shapes,
    )
    package = tmp_path / "radial.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    start = 4 if dynamic else 3
    for count in (7, 11, 9) if dynamic else (values[1].shape[0],):
        if dynamic:
            values = (
                values[0],
                (
                    torch.randn(count, 4, dtype=torch.float64, device="cuda") * 0.1
                ).requires_grad_(),
                (
                    torch.randn(count, 3, dtype=torch.float64, device="cuda") * 0.1
                ).requires_grad_(),
                torch.randint(3, (2, count), device="cuda"),
                *values[start:],
            )
        for _ in range(3):
            expected = operation(*values)
            records.append(
                {
                    "inputs": tuple(v.detach().cpu() for v in values),
                    "outputs": tuple(v.detach().cpu() for v in expected),
                }
            )
            parameters = tuple(
                (v - 0.01 * g).detach().requires_grad_()
                for v, g in zip(values[start:], expected[4:])
            )
            values = (*values[:start], *parameters)
    data = tmp_path / "values.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual), row['outputs'], atol=3e-9, rtol=3e-9)
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
