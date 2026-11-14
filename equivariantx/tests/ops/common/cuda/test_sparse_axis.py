"""Dynamic sparse axes and independently loaded joint AOTI gradients."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.torch.sparse_axis import reference
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native sparse axes require CUDA"
)


def test_forward_ad():
    from equivariantx.ops.common.cuda.sparse_axis import evaluate
    from equivariantx.tests.utils.testing import check_forward_ad

    indices = torch.tensor([[0, 1, 2], [2, 4, 4]], device="cuda")
    x = torch.randn(2, 3, 30, device="cuda", dtype=torch.float64).requires_grad_()
    w = torch.randn(2, 1, 2, 3, device="cuda", dtype=torch.float64).requires_grad_()
    check_forward_ad(
        lambda x, w: evaluate(x, indices, w, 2, 3),
        lambda x, w: reference(x, indices, w, 2, 3),
        (x, w),
    )


def test_dynamic_inductor():
    from equivariantx.ops.common.cuda.sparse_axis import evaluate

    compiled = torch.compile(evaluate, dynamic=True, fullgraph=True)
    indices = torch.tensor([[0, 1, 2], [2, 4, 4], [3, 0, 1]], device="cuda")
    for rows in (2, 5):
        x = (torch.randn(rows, 3, 60, device="cuda", dtype=torch.float64) * 0.1)[
            ..., ::2
        ].requires_grad_()
        w = (
            (torch.randn(rows, 1, 3, 3, device="cuda", dtype=torch.float64) * 0.1)
            .transpose(-1, -2)
            .requires_grad_()
        )
        actual, expected = compiled(x, indices, w, 2, 3), reference(x, indices, w, 2, 3)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), (x, w)),
            torch.autograd.grad(expected.sin().sum(), (x, w)),
            atol=3e-10,
            rtol=3e-10,
        )


def test_joint_aoti(tmp_path):
    from equivariantx.ops.common.cuda.sparse_axis import evaluate

    indices = torch.tensor([[0, 1, 2], [2, 4, 4]], device="cuda")
    x = (
        torch.randn(2, 3, 30, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    w = (
        torch.randn(2, 1, 2, 3, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()

    def operation(x, w, ordinary=False):
        y = (reference if ordinary else evaluate)(x, indices, w, 2, 3)
        energy = y.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(lambda x, w: operation(x, w))(x, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, (x.detach(), w.detach()), strict=True)
    package = tmp_path / "sparse.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(x, w, True)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in (x, w)),
                "outputs": tuple(v.detach().cpu() for v in expected),
            }
        )
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
        x = (x.detach() * 0.9 + 0.01).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
import equivariantx
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
