"""Default indexed frames support nonconstant force-loss AOTI and raw duals."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import o2
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Indexed native frames require CUDA"
)


def problem(inverse):
    frame = o2.LocalFrame("2x0e+2x1o+2x1e+2x2e", mmax=1).cuda().double()
    wigner = o2.WignerD(2, 2, backend="torch", method="recursive").cuda().double()
    width = frame.output_dim if inverse else frame.input_dim
    values = tuple(
        (
            torch.randn(*shape, device="cuda", dtype=torch.float64) * 0.1 + shift
        ).requires_grad_()
        for shape, shift in (
            ((5 if inverse else 3, width), 0),
            ((5, 3), 0.5),
            ((width,), 0.5),
        )
    )
    index = torch.tensor([0, 1, 1, 2, 0], device="cuda")

    def prediction(x, vectors, weight, backend="cuda"):
        rotation = wigner.forward_packed(vectors)
        features = x * weight
        return (
            frame.scatter(features, rotation, index, 3, backend=backend)
            if inverse
            else frame.gather(features, rotation, index, backend=backend)
        )

    def training(x, vectors, weight, backend="cuda"):
        output = prediction(x, vectors, weight, backend)
        readout = torch.linspace(
            0.2, 1.0, output.shape[-1], device=output.device, dtype=output.dtype
        )
        energy = 1 + (output.sin() * readout).mean()
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = energy.square() + (force - 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    return values, prediction, training


@pytest.mark.parametrize("inverse", [False, True])
def test_raw_forward_ad(inverse):
    values, prediction, _ = problem(inverse)
    directions = tuple(torch.randn_like(value) * 0.1 for value in values)
    expected = torch.func.jvp(
        lambda *v: prediction(*v, backend="torch"), values, directions
    )
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(value, tangent)
            for value, tangent in zip(values, directions)
        )
        output = torch.autograd.forward_ad.unpack_dual(prediction(*duals))
        torch.testing.assert_close(
            (output.primal, output.tangent), expected, atol=3e-9, rtol=3e-9
        )


@pytest.mark.parametrize("inverse", [False, True])
def test_default_force_loss_package(inverse, tmp_path):
    values, _, training = problem(inverse)
    expected = training(*values, backend="torch")
    assert expected[1].abs().max() > 1e-7
    torch.testing.assert_close(training(*values), expected, atol=3e-9, rtol=3e-9)
    graph = make_fx(lambda *v: training(*v))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any(
        "eqx.indexed_frame" in str(node.target) for node in graph.graph.nodes
    )
    exported = torch.export.export(
        graph, tuple(value.detach() for value in values), strict=True
    )
    package = tmp_path / "frame.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = training(*values, backend="torch")
        records.append(
            {
                "inputs": tuple(value.detach().cpu() for value in values),
                "output": tuple(value.detach().cpu() for value in expected),
            }
        )
        x, vectors, weight = values
        values = (
            (x.detach() * 0.95 + 0.01).requires_grad_(),
            (vectors.detach() * 0.9 + 0.02).requires_grad_(),
            (weight.detach() - 0.01 * expected[-1]).requires_grad_(),
        )
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad(): actual = run(*(value.cuda() for value in case['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), case['output'], atol=3e-9, rtol=3e-9)
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
