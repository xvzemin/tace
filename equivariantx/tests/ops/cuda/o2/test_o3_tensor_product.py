"""Native aligned readout inside independently loaded force-loss packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import o2
from equivariantx.ops import config
from equivariantx.ops.torch.o2.o3_tensor_product import reference
from equivariantx.tests.ops.torch.o2.test_o3_tensor_product import training_problem

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native aligned readouts require CUDA"
)


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
def test_joint_aoti_without_eqx(mode, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ALIGNED_PRODUCT_INLINE", False)
    run, values = training_problem(mode, "cuda", "cuda")
    graph = make_fx(lambda *args: run(*args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(v.detach() for v in values), strict=True
    )
    package = tmp_path / "aligned.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    x, r, w = values
    for _ in range(3):
        expected = run(x, r, w, True)
        records.append(
            dict(
                inputs=tuple(v.detach().cpu() for v in (x, r, w)),
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


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
def test_native_dynamic_inductor(mode, monkeypatch):
    monkeypatch.setattr(config, "ALIGNED_PRODUCT_INLINE", False)
    layer = (
        o2.O3TensorProduct(
            "2x1o",
            "1o",
            "2x0e+2x1e+2x2e",
            [(0, 0, i, mode, True) for i in range(3)],
            internal_weights=False,
            shared_weights=False,
            backend="cuda",
        )
        .cuda()
        .double()
    )
    compiled = torch.compile(layer.forward_local, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = (
            torch.randn(
                count,
                layer.local_frame_in.output_dim,
                device="cuda",
                dtype=torch.float64,
            )
            * 0.1
        )
        w = torch.randn(1, layer.weight_numel, device="cuda", dtype=torch.float64) * 0.1
        h = torch.randn(count, 1, device="cuda", dtype=torch.float64) * 0.1 + 1
        torch.testing.assert_close(
            compiled(x, w, h), reference(layer, x, w, h), atol=3e-10, rtol=3e-10
        )
