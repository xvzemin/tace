"""Spatial linear maps and normalized gated activations."""

import pytest
import torch
from e3nn import o3

from eqx import o3 as eqx_o3


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


@pytest.mark.parametrize(
    "irreps_in,irreps_out", [("", "2x0e"), ("2x1o", "3x0e"), ("0x0e+2x1o", "0x1o+0e")]
)
def test_indexed_linear_disconnected(irreps_in, irreps_out):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = o3.Linear(
        irreps_in, irreps_out, internal_weights=False, shared_weights=False
    )
    module = eqx_o3.ElementLinear(linear)
    x = torch.randn(3, linear.irreps_in.dim, device=device, requires_grad=True)
    w = torch.randn(2, linear.weight_numel, device=device, requires_grad=True)
    y = module(x, w, torch.tensor([0, 1, 0], device=device))
    assert y.shape == (3, linear.irreps_out.dim)
    assert not y.count_nonzero()
    for gradient in torch.autograd.grad(y.sum(), (x, w)):
        assert not gradient.count_nonzero()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_experts", [1, 2])
@pytest.mark.parametrize("num_nodes", [0, 5])
def test_indexed_linear_derivatives(device, num_experts, num_nodes, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o3.Linear(
        "2x0e+0e+2x1o+1e",
        "3x0e+2x1o+1o+2o",
        internal_weights=False,
        shared_weights=False,
    )
    cls = eqx_o3.ElementLinear if num_experts == 1 else eqx_o3.MoEElementLinear
    kwargs = {} if num_experts == 1 else {"num_experts": num_experts}
    module = cls(linear, **kwargs)
    reference = cls(linear, backend="torch", **kwargs)
    x = torch.randn(module.irreps_in.dim, num_nodes, device=device).T.requires_grad_()
    shape = (
        (3, linear.weight_numel)
        if num_experts == 1
        else (3, num_experts, linear.weight_numel)
    )
    weight = torch.randn(*shape, device=device, requires_grad=True)
    types = (torch.arange(2 * num_nodes, device=device) % 3)[::2]
    actual, expected = module(x, weight, types), reference(x, weight, types)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((y.sin() * seed).sum(), (x, weight), create_graph=True)
            for y in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) for grads in gradients
        ]
    assert not module.state_dict()


@pytest.mark.parametrize("experts", [1, 2])
def test_indexed_linear_compile_and_capture(experts, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o3.Linear(
        "2x0e+2x1o", "3x0e+2x1o", internal_weights=False, shared_weights=False
    )
    module = eqx_o3.MoEElementLinear(linear, experts)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, module.irreps_in.dim, device="cuda", requires_grad=True)
        weight = torch.randn(
            2, experts, linear.weight_numel, device="cuda", requires_grad=True
        )
        types = torch.arange(count, device="cuda") % 2
        actual, expected = compiled(x, weight, types), module(x, weight, types)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (x, weight)),
            torch.autograd.grad(expected.square().sum(), (x, weight)),
        ):
            torch.testing.assert_close(a, b)
    x = torch.randn(7, module.irreps_in.dim, device="cuda", requires_grad=True)
    weight = torch.randn(
        2, experts, linear.weight_numel, device="cuda", requires_grad=True
    )
    types = torch.arange(7, device="cuda") % 2

    def evaluate():
        y = module(x, weight, types)
        first = torch.autograd.grad(y.sin().sum(), (x, weight), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), (x, weight))
        return y, *first, *second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = evaluate()
    with torch.no_grad():
        x.normal_()
        weight.normal_()
        types.random_(2)
    graph.replay()
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
