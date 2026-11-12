"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o3 as eqx_o3

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.parametrize(
    "irreps_in,irreps_out", [("", "2x0e"), ("2x1o", "3x0e"), ("0x0e+2x1o", "0x1o+0e")]
)
def test_indexed_linear_disconnected(irreps_in, irreps_out):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = o3.Linear(
        irreps_in, irreps_out, internal_weights=False, shared_weights=False
    )
    module = eqx_o3.ElementLinear(linear, backend="cuda")
    x = torch.randn(3, linear.irreps_in.dim, device=device, requires_grad=True)
    w = torch.randn(2, linear.weight_numel, device=device, requires_grad=True)
    y = module(x, w, torch.tensor([0, 1, 0], device=device))
    assert y.shape == (3, linear.irreps_out.dim)
    assert not y.count_nonzero()
    for gradient in torch.autograd.grad(y.sum(), (x, w)):
        assert not gradient.count_nonzero()


@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_experts", [1, 2])
@pytest.mark.parametrize("num_nodes", [0, 5, 35])
@pytest.mark.parametrize("expand_weights", [False, True])
def test_indexed_linear_derivatives(
    mode, device, num_experts, num_nodes, expand_weights, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o3.Linear(
        "2x0e+0e+2x1o+1e",
        "3x0e+2x1o+1o+2o",
        internal_weights=False,
        shared_weights=False,
    )
    cls = eqx_o3.ElementLinear if num_experts == 1 else eqx_o3.MoEElementLinear
    kwargs = {"element_mode": mode}
    if num_experts != 1:
        kwargs["num_experts"] = num_experts
    module = cls(linear, backend="cuda", **kwargs)
    reference = cls(linear, **kwargs)
    assert reference.backend == "torch"
    assert not reference.expand_weights
    reference.expand_weights = expand_weights
    x = torch.randn(module.irreps_in.dim, num_nodes, device=device).T.requires_grad_()
    shape = (
        (3, linear.weight_numel)
        if num_experts == 1
        else (3, num_experts, linear.weight_numel)
    )
    if mode == "pair":
        shape = (3, 2, *shape[1:])
    weight = torch.randn(*shape, device=device, requires_grad=True)
    types = (torch.arange(2 * num_nodes, device=device) % 3)[::2]
    target = torch.arange(num_nodes, device=device) % 2 if mode == "pair" else None
    actual, expected = (
        module(x, weight, types, target),
        reference(x, weight, types, target),
    )
    if num_experts == 1 and num_nodes:
        selected = weight[types, target] if mode == "pair" else weight[types]
        torch.testing.assert_close(actual, linear(x, selected), atol=1e-12, rtol=1e-12)
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


@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("experts", [1, 2])
def test_indexed_linear_capture(mode, experts, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o3.Linear(
        "2x0e+2x1o", "3x0e+2x1o", internal_weights=False, shared_weights=False
    )
    module = eqx_o3.MoEElementLinear(linear, experts, backend="cuda", element_mode=mode)
    shape = (
        (2, 3, experts, linear.weight_numel)
        if mode == "pair"
        else (2, experts, linear.weight_numel)
    )
    x = torch.randn(7, module.irreps_in.dim, device="cuda", requires_grad=True)
    weight = torch.randn(*shape, device="cuda", requires_grad=True)
    types = torch.arange(7, device="cuda") % 2
    target = torch.arange(7, device="cuda") % 3 if mode == "pair" else None

    def evaluate():
        y = module(x, weight, types, target)
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
        if target is not None:
            target.random_(3)
    graph.replay()
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
