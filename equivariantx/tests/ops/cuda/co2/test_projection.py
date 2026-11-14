"""Native plane/detracing packages and dynamic public projection."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.ops import config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native projections require CUDA"
)


@pytest.mark.parametrize("rank", [2, 3])
def test_joint_aoti_without_eqx(monkeypatch, tmp_path, rank):
    monkeypatch.setattr(config, "TENSOR_AXIS_INLINE", False)
    layers = {
        backend: (
            co2.PlaneProjector(rank, backend=backend).cuda(),
            co2.PlanarDetracer(
                rank, backend=backend, dtype=torch.float64, device="cuda"
            ),
        )
        for backend in ("torch", "cuda")
    }

    def operation(x, vectors, weight, ordinary=False):
        backend = "torch" if ordinary else "cuda"
        plane = co2.plane_projector(
            torch.nn.functional.normalize(vectors, dim=-1), backend=backend
        )
        project, detrace = layers[backend]
        y = detrace(project(x * weight, plane), plane)
        energy = y.sin().sum() * 0.1
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = (force - 0.01).square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    x = (
        torch.randn(2, 3, 3**rank, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    vectors = (
        torch.randn(2, 1, 3, device="cuda", dtype=torch.float64) + 0.5
    ).requires_grad_()
    weight = (
        torch.randn(3, 1, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    inputs = x, vectors, weight
    graph = make_fx(lambda *v: operation(*v))(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(v.detach() for v in inputs), strict=True
    )
    package = tmp_path / "projection.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(x, vectors, weight, True)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in (x, vectors, weight)),
                "output": tuple(v.detach().cpu() for v in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
        vectors = (vectors.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad(): actual=run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),record['output'],atol=3e-9,rtol=3e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("rank", [2, 3])
def test_dynamic_inductor(monkeypatch, rank):
    monkeypatch.setattr(config, "TENSOR_AXIS_INLINE", False)
    project = co2.PlaneProjector(rank, backend="cuda").cuda()
    detrace = co2.PlanarDetracer(
        rank, backend="cuda", dtype=torch.float64, device="cuda"
    )

    def operation(x, vectors):
        plane = co2.plane_projector(
            torch.nn.functional.normalize(vectors, dim=-1), backend="cuda"
        )
        return detrace(project(x, plane), plane)

    compiled = torch.compile(operation, dynamic=True, fullgraph=True)
    for rows in (2, 5):
        x = (
            torch.randn(rows, 3, 2 * 3**rank, device="cuda", dtype=torch.float64) * 0.2
        )[..., ::2].requires_grad_()
        vectors = (
            torch.randn(rows, 1, 3, device="cuda", dtype=torch.float64) + 0.5
        ).requires_grad_()
        actual, expected = compiled(x, vectors), operation(x, vectors)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), (x, vectors)),
            torch.autograd.grad(expected.sin().sum(), (x, vectors)),
            atol=3e-9,
            rtol=3e-9,
        )
