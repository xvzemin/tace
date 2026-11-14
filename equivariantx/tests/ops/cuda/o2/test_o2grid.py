"""Native grid force gradients load from AOTI independently of EQX."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.o2.o2grid import O2Grid
from equivariantx.ops import config
from equivariantx.ops.torch.o2 import o2grid as kernels
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native grids require CUDA"
)


@pytest.mark.parametrize("time_odd", [False, True])
def test_joint_aoti(time_odd, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "O2_GRID_INLINE", False)
    layer = O2Grid(
        "0ee+0oe+2x1me" + ("+0eo+0oo+2x1mo" if time_odd else ""),
        backend="cuda",
        dtype=torch.float64,
        device="cuda",
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


@pytest.mark.parametrize("time_odd", [False, True])
def test_native_dynamic_inductor(time_odd, monkeypatch):
    monkeypatch.setattr(config, "O2_GRID_INLINE", False)
    layer = O2Grid(
        "0ee+0oe+2x1me" + ("+0eo+0oo+2x1mo" if time_odd else ""),
        backend="cuda",
        dtype=torch.float64,
        device="cuda",
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


@pytest.mark.parametrize("mmax", [0, 8, 32])
def test_native_wide_noncontiguous(mmax, monkeypatch):
    monkeypatch.setattr(config, "O2_GRID_INLINE", False)
    irreps = "+".join(
        f"2x0e{time}+2x0o{time}" + "".join(f"+4x{m}m{time}" for m in range(1, mmax + 1))
        for time in ("e", "o")
    )
    layer = O2Grid(irreps, backend="cuda", dtype=torch.float32).cuda().double()
    for count in (0, 3):
        x = (
            torch.randn(layer.dim, count, 2, device="cuda", dtype=torch.float64)
            .movedim(0, -1)
            .requires_grad_()
        )
        actual = layer(x)
        expected = kernels.to_grid_reference(layer, x)
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        torch.testing.assert_close(layer.from_grid(actual), x, atol=3e-12, rtol=3e-12)
        if count:
            torch.testing.assert_close(
                torch.autograd.grad(actual.sin().sum(), x)[0],
                torch.autograd.grad(expected.sin().sum(), x)[0],
                atol=3e-12,
                rtol=3e-12,
            )
