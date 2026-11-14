"""Native Chebyshev operand derivatives and independent AOTI loading."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_values_and_derivatives(dtype):
    from equivariantx.ops.cuda.nn.activation import chebyshev
    from equivariantx.ops.torch.nn.activation import Chebyshev, reference

    x = (torch.randn(3, 2, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    coefficients = (torch.randn(9, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    torch.testing.assert_close(
        chebyshev(x[..., 0], coefficients), reference(x[..., 0], coefficients)
    )
    assert chebyshev(x[:0], coefficients).shape == x[:0].shape
    with pytest.raises(TypeError):
        chebyshev(x.half(), coefficients.half())
    if dtype == torch.float64:
        for fn in (Chebyshev.apply, chebyshev):
            assert torch.autograd.gradcheck(fn, (x, coefficients))
            assert torch.autograd.gradgradcheck(fn, (x, coefficients))


def test_joint_aoti_without_eqx(tmp_path):
    from equivariantx.ops.cuda.nn.activation import chebyshev
    from equivariantx.ops.torch.nn.activation import reference

    x = (torch.randn(5, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
    weight = (torch.randn(9, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()

    def operation(x, weight, ordinary=False):
        value = (reference if ordinary else chebyshev)(x, weight)
        energy = value.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = (force - 0.2).square().mean() + energy.square() * 0.01
        return loss, force, torch.autograd.grad(loss, weight)[0]

    graph = make_fx(lambda x, w: operation(x, w))(x, weight)
    exported = torch.export.export(graph, (x.detach(), weight.detach()), strict=True)
    package = tmp_path / "activation.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = operation(x, weight, True)
        records.append(
            {
                "inputs": (x.detach().cpu(), weight.detach().cpu()),
                "output": tuple(v.detach().cpu() for v in expected),
            }
        )
        weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
        x = (x.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(x.cuda() for x in record['inputs']))
    torch.testing.assert_close(tuple(x.cpu() for x in actual), record['output'], atol=2e-10, rtol=2e-10)
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


def test_forward_ad():
    from equivariantx.nn.activation import PolynomialActivation
    from equivariantx.tests.utils.testing import check_forward_ad

    layers = {
        backend: PolynomialActivation("silu", degree=8, backend=backend).double().cuda()
        for backend in ("torch", "cuda")
    }
    x = (torch.randn(3, 2, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
    check_forward_ad(layers["cuda"], layers["torch"], (x,))
