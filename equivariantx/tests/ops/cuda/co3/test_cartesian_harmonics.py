"""Exact native harmonic derivatives and independent AOTI execution."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 11])
def test_native_values(dtype, rows):
    from equivariantx.ops.cuda.co3.cartesian_harmonics import low_order
    from equivariantx.ops.torch.co3.cartesian_harmonics import low_order as reference

    value = torch.randn(rows, 3, 2, device="cuda", dtype=dtype)[..., 0]
    orders = [2, 0, 1, 2]
    actual = low_order(value, orders, "component")
    torch.testing.assert_close(actual, reference(value, orders, "component"))
    with pytest.raises(TypeError):
        low_order(value.half(), orders, "component")


@pytest.mark.parametrize(
    "degree, backend",
    [(2, "torch"), (2, "cuda"), (2, "native"), (4, "torch"), (4, "cuda")],
)
def test_joint_aoti(tmp_path, backend, degree):
    from equivariantx.ops.torch.co3.cartesian_harmonics import reference

    layer = (
        co3.CartesianHarmonics(
            list(range(degree + 1)),
            True,
            "norm",
            backend="cuda" if backend == "native" else backend,
        )
        .cuda()
        .double()
    )
    vectors = (
        torch.randn(5, 3, device="cuda", dtype=torch.float64) * 0.1 + 0.5
    ).requires_grad_()
    weight = (
        torch.randn(layer.irreps_out.dim, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()

    def operation(vectors, weight, ordinary=False):
        if backend == "native" and not ordinary:
            from equivariantx.ops.cuda.co3.cartesian_harmonics import low_order

            normalized = torch.nn.functional.normalize(vectors, dim=-1)
            harmonics = low_order(normalized, list(range(degree + 1)), "norm")
        else:
            harmonics = reference(layer, vectors) if ordinary else layer(vectors)
        energy = (harmonics * weight).sin().sum()
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = (force - 0.2).square().mean() + energy.square() * 0.01
        return loss, force, torch.autograd.grad(loss, weight)[0]

    graph = make_fx(lambda x, w: operation(x, w))(vectors, weight)
    if backend != "native":
        assert not any(
            str(node.target).startswith("eqx.") for node in graph.graph.nodes
        )
    inputs = vectors.detach(), weight.detach()
    exported = torch.export.export(graph, inputs, strict=True)
    package = tmp_path / "harmonics.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(vectors, weight, True)
        records.append(
            {
                "inputs": (vectors.detach().cpu(), weight.detach().cpu()),
                "output": tuple(value.detach().cpu() for value in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
        vectors = (vectors.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), record['output'], atol=2e-10, rtol=2e-10)
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
    from equivariantx import co3
    from equivariantx.tests.utils.testing import check_forward_ad

    layers = {
        backend: co3.CartesianHarmonics([0, 1, 2, 3], normalize=True, backend=backend)
        .double()
        .cuda()
        for backend in ("torch", "cuda")
    }
    x = torch.randn(3, 3, device="cuda", dtype=torch.float64).requires_grad_()
    check_forward_ad(layers["cuda"], layers["torch"], (x,))
