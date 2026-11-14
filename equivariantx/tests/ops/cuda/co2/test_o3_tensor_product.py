"""Native angular kernels in independently loaded joint training artifacts."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import config
from equivariantx.tests.ops.torch.co2.test_o3_tensor_product import training_problem
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


def test_joint_aoti(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "ICTP_INLINE", False)
    monkeypatch.setattr(config, "SPHERICAL_INLINE", False)
    operation, values = training_problem("cuda", "cuda")
    graph = make_fx(operation)(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(v.detach() for v in values), strict=True
    )
    package = tmp_path / "module.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    x, y, w = values
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        expected = operation(x, y, w)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in (x, y, w)),
                "outputs": tuple(v.detach().cpu() for v in expected),
            }
        )
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
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
