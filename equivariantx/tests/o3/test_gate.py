"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import o3 as eqx_o3

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.parametrize("num_nodes", [0, 5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("gate_scalars", [False, True])
def test_gate_derivatives(num_nodes, device, gate_scalars, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn.nn import Gate

    args = (
        "2x0e+0o+0e",
        [torch.nn.SiLU(), torch.nn.Tanh(), None],
        "2x0e+2x0o",
        [torch.nn.Sigmoid(), torch.nn.Tanh()],
        "2x0e+1o+3e" if gate_scalars else "2x1o+1e+3e",
    )
    module = eqx_o3.Gate(*args).to(device)
    reference = Gate(*args).to(device)
    module.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(module.irreps_in.dim, num_nodes, device=device).T.requires_grad_()
    actual, expected = module(x), reference(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.autograd.grad((y.sin() * seed).sum(), x, create_graph=True)[0]
            for y in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("scalar_only", [False, True])
def test_gate_compile(scalar_only, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn.nn import Gate

    args = (
        ("2x0e", [torch.nn.SiLU()], "", [], "")
        if scalar_only
        else ("", [], "2x0e", [torch.nn.Sigmoid()], "1o+2e")
    )
    module = eqx_o3.Gate(*args).cuda()
    reference = Gate(*args).cuda()
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(
            2, count, module.irreps_in.dim, device="cuda", requires_grad=True
        )
        actual, expected = compiled(x), reference(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=1e-12,
            rtol=1e-12,
        )


def test_gate_fallback(double_precision):
    from e3nn.nn import Gate

    for backend, act in (("torch", torch.nn.SiLU()), ("cuda", torch.nn.ReLU())):
        args = ("2x0e", [act], "0e", [None], "1o")
        module = eqx_o3.Gate(*args, backend=backend)
        if backend == "cuda":
            assert module.metadata is None
        x = torch.randn(4, module.irreps_in.dim, requires_grad=True)
        actual, expected = module(x), Gate(*args)(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=0,
            rtol=0,
        )
