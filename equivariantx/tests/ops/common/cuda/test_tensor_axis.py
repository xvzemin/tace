"""Native tensor-axis dynamic compilation and independent AOTI loading."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.cuda.tensor_axis import evaluate
from equivariantx.ops.common.torch.tensor_axis import reference
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native tensor axes require CUDA"
)


def test_dynamic_inductor():
    run = torch.compile(evaluate, fullgraph=True, dynamic=True)
    for rows in (2, 5):
        x = (torch.randn(rows, 3, 54, device="cuda", dtype=torch.float64) * 0.2)[
            ..., ::2
        ].requires_grad_()
        matrix = (
            torch.randn(rows, 1, 3, 3, device="cuda", dtype=torch.float64).transpose(
                -1, -2
            )
            * 0.2
        ).requires_grad_()
        actual, expected = run(x, matrix, 3, 3), reference(x, matrix, 3, 3)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), (x, matrix)),
            torch.autograd.grad(expected.sin().sum(), (x, matrix)),
            atol=3e-10,
            rtol=3e-10,
        )


def test_joint_aoti(tmp_path):
    x = (
        torch.randn(2, 3, 27, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()
    matrix = (
        torch.randn(2, 1, 3, 3, device="cuda", dtype=torch.float64) * 0.2
    ).requires_grad_()

    def operation(x, matrix, ordinary=False):
        y = (reference if ordinary else evaluate)(x, matrix, 3, 3)
        energy = y.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = (force - 0.2).square().mean() + energy.square() * 0.01
        return loss, force, torch.autograd.grad(loss, matrix)[0]

    graph = make_fx(lambda x, m: operation(x, m))(x, matrix)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(graph, (x.detach(), matrix.detach()), strict=True)
    package = tmp_path / "tensor-axis.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(x, matrix, True)
        records.append(
            {
                "inputs": tuple(v.detach().cpu() for v in (x, matrix)),
                "output": tuple(v.detach().cpu() for v in expected),
            }
        )
        matrix = (matrix - 0.01 * expected[-1]).detach().requires_grad_()
        x = (x.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
import equivariantx
run=torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad(): actual=run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),record['output'],atol=3e-10,rtol=3e-10)
assert 'equivariantx' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", package_script(script), str(package), str(data)],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_forward_ad():
    from equivariantx.tests.utils.testing import check_forward_ad

    x = torch.randn(2, 3, 27, device="cuda", dtype=torch.float64).requires_grad_()
    matrix = torch.randn(
        2, 1, 3, 3, device="cuda", dtype=torch.float64
    ).requires_grad_()
    check_forward_ad(
        lambda x, m: evaluate(x, m, 3, 3),
        lambda x, m: reference(x, m, 3, 3),
        (x, matrix),
    )
