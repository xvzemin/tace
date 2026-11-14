"""Native sparse bilinear recursion, dynamic compilation and standalone AOTI."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.torch.sparse_bilinear import reference
from equivariantx.tests.ops.common.torch.test_sparse_bilinear import (
    check_coefficient_forward_ad,
    check_derivatives,
    coefficient_tables,
)
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("trainable", [False, True])
def test_coefficient_forward_ad(role, trainable):
    from equivariantx.ops.common.cuda.sparse_bilinear import evaluate

    check_coefficient_forward_ad(evaluate, "cuda", role, trainable)


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("dense", [False, True])
def test_derivatives(role, dense):
    from equivariantx.ops.common.cuda.sparse_bilinear import evaluate

    check_derivatives(evaluate, "cuda", role, dense)


def test_dynamic_inductor():
    from equivariantx.ops.common.cuda.sparse_bilinear import evaluate

    _, tables = coefficient_tables("cuda")

    def operation(x, y):
        return evaluate(x, y, tables)

    run = torch.compile(operation, fullgraph=True, dynamic=True)
    for rows in (2, 5, 0):
        x = torch.randn(rows, 3, device="cuda", dtype=torch.float64, requires_grad=True)
        y = torch.randn(1, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        actual, expected = run(x, y), reference(x, y, tables)
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), (x, y)),
            torch.autograd.grad(expected.sin().sum(), (x, y)),
            atol=3e-11,
            rtol=3e-11,
        )


def test_joint_aoti(tmp_path):
    from equivariantx.ops.common.cuda.sparse_bilinear import evaluate

    _, tables = coefficient_tables("cuda")
    x = (torch.randn(2, 3, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
    y = (torch.randn(2, 4, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
    w = torch.randn(3, device="cuda", dtype=torch.float64, requires_grad=True)

    def operation(x, y, w, ordinary=False):
        z = (reference if ordinary else evaluate)(x * w, y, tables)
        energy = 1 + z.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(lambda x, y, w: operation(x, y, w))(x, y, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(v.detach() for v in (x, y, w)), strict=True
    )
    package = tmp_path / "bilinear.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(x, y, w, True)
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
import equivariantx
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad(): actual=run(*(v.cuda() for v in row['inputs']))
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


@pytest.mark.parametrize("dimensions", [(0, 3, 2), (2, 0, 3), (2, 3, 0)])
def test_empty_components(dimensions):
    from equivariantx.ops.common.cuda.sparse_bilinear import evaluate
    from equivariantx.ops.common.sparse_coefficients import bilinear_tables

    tables = bilinear_tables(dimensions, (), device="cuda")
    x = torch.empty(
        2, dimensions[0], dtype=torch.float64, device="cuda", requires_grad=True
    )
    y = torch.empty(
        1, dimensions[1], dtype=torch.float64, device="cuda", requires_grad=True
    )
    z = evaluate(x, y, tables)
    torch.testing.assert_close(
        z, torch.zeros(2, dimensions[2], dtype=torch.float64, device="cuda")
    )
    dx, dy = torch.autograd.grad(z.sum(), (x, y))
    torch.testing.assert_close(dx, torch.zeros_like(x))
    torch.testing.assert_close(dy, torch.zeros_like(y))
