"""Indexed spatial linear maps, including element and expert coefficients."""

import pytest
import torch
from e3nn import o3

from eqx import o3 as eqx_o3
from tace.models.linear import e3nnElementLinear, e3nnMoEElementLinear, enable_lora
from tace.utils.env import EQX_KERNELS


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
@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tace_indexed_linear(monkeypatch, experts, matrix, lora, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.set_default_dtype(dtype)
    cls = e3nnElementLinear if experts == 1 else e3nnMoEElementLinear
    module = cls(
        "4x0e+2x0e+4x1o+2x1e",
        "6x0e+4x1o+2x1o+2x2o",
        num_elements=3,
        bias=True,
        use_matrix_weight=matrix,
        **({} if experts == 1 else {"num_experts": experts}),
    ).cuda()
    if lora:
        enable_lora(module, r=2, freeze_base=False)
        with torch.no_grad():
            for parameter in module.lora_B:
                parameter.normal_()
    with torch.no_grad():
        module.bias.normal_()
    attrs = torch.rand(5, 3, device="cuda", requires_grad=True)
    x = torch.randn(5, module.irreps_in.dim, device="cuda", requires_grad=True)
    # A supplied element index takes precedence over attrs.argmax for weights.
    types = torch.tensor([2, 0, 1, 2, 1], device="cuda")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    outputs, derivatives = [], []
    inputs = (x, attrs, *module.parameters())
    for enabled in (False, True):
        monkeypatch.setitem(EQX_KERNELS, "linear", enabled)
        output = module(x, attrs, types)
        first = torch.autograd.grad(output.square().mean(), x, create_graph=True)[0]
        derivatives.append(
            torch.autograd.grad(
                output.square().mean() + first.square().mean(),
                inputs,
                allow_unused=True,
            )
        )
        outputs.append(output)
    tol = 3e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(*outputs, atol=tol, rtol=tol)
    for actual, expected in zip(*derivatives):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    if dtype == torch.float64 and matrix == "1" and lora:
        compiled = torch.compile(
            module, backend="aot_eager", fullgraph=True, dynamic=True
        )
        output = compiled(x, attrs, types)
        torch.testing.assert_close(output, outputs[1], atol=tol, rtol=tol)
        torch.testing.assert_close(
            torch.autograd.grad(output.square().mean(), x)[0],
            torch.autograd.grad(module(x, attrs, types).square().mean(), x)[0],
            atol=tol,
            rtol=tol,
        )


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
