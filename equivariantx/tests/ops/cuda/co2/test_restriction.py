"""Native restriction and transverse maps in independently loaded force-loss packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2
from equivariantx.co2.restriction import transverse_quarter_turn
from equivariantx.ops import config
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native restriction maps require CUDA"
)


@pytest.mark.parametrize("rank", [2, 4])
def test_joint_aoti(rank, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESTRICTION_INLINE", False)
    monkeypatch.setattr(config, "SYMMETRIC_INLINE", False)
    layers = {
        backend: (
            co2.Restriction(rank, backend=backend).double().cuda(),
            co2.TransverseProjector(rank, backend=backend).double().cuda(),
        )
        for backend in ("torch", "cuda")
    }

    def operation(x, v, w, ordinary=False):
        backend = "torch" if ordinary else "cuda"
        layer, projector = layers[backend]
        n = torch.nn.functional.normalize(v, dim=-1)
        parts = tuple(p.sin() for p in layer(x * w, n))
        y = layer.inverse(parts, n) + projector(x * w, n)
        y = y + transverse_quarter_turn(parts[-1], n, backend=backend)
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    x = (
        torch.randn(2, 2, 3**rank, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    v = (
        torch.randn(2, 1, 3, device="cuda", dtype=torch.float64) + 0.5
    ).requires_grad_()
    w = (torch.randn(2, 1, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
    values = x, v, w
    graph = make_fx(lambda *args: operation(*args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(t.detach() for t in values), strict=True
    )
    package = tmp_path / "restriction.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        expected = operation(x, v, w, True)
        records.append(
            {
                "inputs": tuple(t.detach().cpu() for t in (x, v, w)),
                "outputs": tuple(t.detach().cpu() for t in expected),
            }
        )
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
        v = (v.detach() * 0.9 + 0.1).requires_grad_()
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
for row in torch.load(sys.argv[2],weights_only=True):
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


def test_native_dynamic_inductor(monkeypatch):
    from equivariantx.ops.torch.co2.restriction import restrict_reference

    monkeypatch.setattr(config, "RESTRICTION_INLINE", False)
    monkeypatch.setattr(config, "SYMMETRIC_INLINE", False)
    layer = co2.Restriction(3, backend="cuda").double().cuda()
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, 2, 27, device="cuda", dtype=torch.float64) * 0.1
        n = torch.nn.functional.normalize(
            torch.randn(count, 1, 3, device="cuda", dtype=torch.float64) + 0.5, dim=-1
        )
        torch.testing.assert_close(
            compiled(x, n), restrict_reference(layer, x, n), atol=3e-10, rtol=3e-10
        )


def test_forward_ad():
    from equivariantx.tests.utils.testing import check_forward_ad

    layers = {
        backend: co2.Restriction(3, backend=backend).double().cuda()
        for backend in ("torch", "cuda")
    }

    def operation(x, v, backend):
        layer = layers[backend]
        n = torch.nn.functional.normalize(v, dim=-1)
        parts = tuple(p.sin() for p in layer(x, n))
        return layer.inverse(parts, n) + transverse_quarter_turn(
            parts[-1], n, backend=backend
        )

    x = torch.randn(2, 2, 27, device="cuda", dtype=torch.float64).requires_grad_()
    v = torch.randn(2, 1, 3, device="cuda", dtype=torch.float64).requires_grad_()
    check_forward_ad(
        lambda x, v: operation(x, v, "cuda"),
        lambda x, v: operation(x, v, "torch"),
        (x, v),
    )
