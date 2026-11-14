"""Native grid force gradients load from AOTI independently of EQX."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.o3.s2grid import S2Grid
from equivariantx.ops import config
from equivariantx.ops.torch.o3 import s2grid as kernels
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native grids require CUDA"
)


@pytest.mark.parametrize("quadrature", ["lebedev", "gauss_legendre", "equiangular"])
def test_joint_aoti(quadrature, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "S2_GRID_INLINE", False)
    layer = S2Grid(
        2, quadrature=quadrature, backend="cuda", dtype=torch.float64, device="cuda"
    )

    def run(x, w, ordinary=False):
        y = (
            kernels.to_grid_reference(layer, x * w)
            if ordinary
            else layer.to_grid(x * w)
        )
        y = (
            kernels.from_grid_reference(layer, y.sin())
            if ordinary
            else layer.from_grid(y.sin())
        )
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    x = (
        torch.randn(2, layer.dim, dtype=torch.float64, device="cuda") * 0.1
    ).requires_grad_()
    w = (
        torch.randn(layer.dim, dtype=torch.float64, device="cuda") * 0.1
    ).requires_grad_()
    graph = make_fx(lambda *args: run(*args))(x, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, (x.detach(), w.detach()), strict=True)
    package = tmp_path / "grid.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = run(x, w, True)
        records.append(
            dict(
                inputs=tuple(v.detach().cpu() for v in (x, w)),
                outputs=tuple(v.detach().cpu() for v in expected),
            )
        )
        w = (w - 0.001 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():actual=run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual),row['outputs'],atol=3e-9,rtol=3e-9)
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


@pytest.mark.parametrize("quadrature", ["lebedev", "gauss_legendre", "equiangular"])
def test_native_dynamic_inductor(quadrature, monkeypatch):
    monkeypatch.setattr(config, "S2_GRID_INLINE", False)
    layer = S2Grid(
        2, quadrature=quadrature, backend="cuda", dtype=torch.float64, device="cuda"
    )

    def run(x):
        return layer.from_grid(layer(x).sin())

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, 2, layer.dim, device="cuda", dtype=torch.float64) * 0.1
        expected = kernels.from_grid_reference(
            layer, kernels.to_grid_reference(layer, x).sin()
        )
        torch.testing.assert_close(compiled(x), expected, atol=3e-10, rtol=3e-10)
