"""Native sparse spherical recurrences in independently loaded force-loss packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.co2 import SphericalCoupling
from equivariantx.ops import config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native spherical maps require CUDA"
)


@pytest.mark.parametrize("method", ["chebyshev", "recurrence"])
@pytest.mark.parametrize("channels", [2, 16])
def test_joint_aoti_without_eqx(method, channels, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPHERICAL_INLINE", False)
    layers = {
        backend: SphericalCoupling(2, 3, 2, method=method, backend=backend)
        .double()
        .cuda()
        for backend in ("torch", "cuda")
    }

    def operation(x, v, w, ordinary=False):
        layer = layers["torch" if ordinary else "cuda"]
        y = layer(x * w, v)
        energy = 1 + y.sin().sum()
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = force.square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    x = (
        torch.randn(2, channels, 5, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    v = (
        torch.randn(2, 1, 3, device="cuda", dtype=torch.float64) + 0.5
    ).requires_grad_()
    w = (
        torch.randn(channels, 1, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    values = x, v, w
    graph = make_fx(lambda *args: operation(*args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(t.detach() for t in values), strict=True
    )
    package = tmp_path / "spherical.pt2"
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
            """
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad():actual=run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual),row['outputs'],atol=3e-9,rtol=3e-9)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("channels", [2, 16])
def test_native_dynamic_inductor(channels, monkeypatch):
    from equivariantx.ops.torch.co2.spherical import reference

    monkeypatch.setattr(config, "SPHERICAL_INLINE", False)
    layer = SphericalCoupling(2, 3, 2, backend="cuda").double().cuda()
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, channels, 5, device="cuda", dtype=torch.float64) * 0.1
        v = torch.randn(count, 1, 3, device="cuda", dtype=torch.float64) + 0.5
        actual = compiled(x, v)
        torch.testing.assert_close(
            actual, reference(layer, x, v, layer.method), atol=3e-10, rtol=3e-10
        )


@pytest.mark.parametrize("channels", [2, 16])
def test_forward_ad(channels):
    from equivariantx.ops.torch.co2.spherical import reference
    from equivariantx.tests.utils.testing import check_forward_ad

    layer = SphericalCoupling(2, 3, 2, backend="cuda").double().cuda()
    x = torch.randn(2, channels, 5, device="cuda", dtype=torch.float64).requires_grad_()
    v = torch.randn(2, 1, 3, device="cuda", dtype=torch.float64).requires_grad_()
    check_forward_ad(layer, lambda x, v: reference(layer, x, v, layer.method), (x, v))
