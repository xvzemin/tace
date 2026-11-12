"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import co3, o2
from equivariantx import o3 as eqx_o3
from equivariantx.utils.module import copy_model

from equivariantx.tests.o2.testing import _transform


@pytest.mark.parametrize("mode", ["single", "pair"])
def test_element_linear_autocast(mode, device):
    linear = o2.Linear(
        "3x0e+3x1m", "2x0e+2x1m", internal_weights=False, shared_weights=False
    ).to(device=device, dtype=torch.float32)
    layer = o2.ElementLinear(linear, backend="torch", element_mode=mode)
    x = torch.randn(
        9,
        linear.irreps_in.dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    shape = (3, 3, linear.weight_numel) if mode == "pair" else (3, linear.weight_numel)
    weight = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    source = torch.arange(9, device=device) % 3
    target = source.roll(1) if mode == "pair" else None
    with pytest.raises(TypeError, match="float32 or float64"):
        layer(x, weight, source, target)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            layer(x.float(), weight, source, target)
    if x.is_cuda:
        layer.backend = "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with pytest.raises(TypeError, match="autocast"):
                layer(x.float(), weight, source, target)


def test_element_linear_parameters():
    linear = o2.Linear("0e", "0e")
    layer = o2.ElementLinear(linear, backend="torch")
    weight = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    source = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 0, 1])
    x = torch.ones(4, 1)
    torch.testing.assert_close(
        layer(x, weight, source, target), torch.arange(1.0, 5.0).reshape(4, 1)
    )
    source = torch.tensor([0] * 16 + [0, 1, 1])
    target = torch.tensor([0] * 16 + [1, 0, 1])
    x = torch.ones(19, 1, requires_grad=True)
    weight.requires_grad_()
    actual, expected = layer(x, weight, source, target), x * weight[source, target]
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, weight)),
        torch.autograd.grad(expected.square().sum(), (x, weight)),
    ):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="target_type"):
        layer(x, weight, source)
    with pytest.raises(ValueError, match="Weights"):
        layer(x, weight.flatten(0, 1), source, target)
    with pytest.raises(ValueError, match="unbiased"):
        o2.ElementLinear(o2.Linear("0e", "0e", biases=True))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("expand_weights", [False, True])
def test_element_linear(backend, device, mode, dtype, expand_weights):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o2.Linear(
        "2x0ee+0ee+2x0oo+3x1me+2x1mo+2x3me",
        "3x0ee+2x0oo+2x1me+1me+2x1mo+2x3me+2me",
        internal_weights=False,
        shared_weights=False,
    ).to(device=device, dtype=dtype)
    module = o2.ElementLinear(linear, backend, element_mode=mode)
    assert not module.expand_weights
    module.expand_weights = expand_weights
    assert o2.ElementLinear(linear).element_mode == "pair"
    assert o2.ElementLinear(linear).backend == "torch"
    shape = (3, 4, linear.weight_numel) if mode == "pair" else (3, linear.weight_numel)
    weight = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
    for count in (0, 7):
        x = torch.randn(
            linear.irreps_in.dim, count, dtype=dtype, device=device
        ).T.requires_grad_()
        source = (torch.arange(2 * count, device=device) % 3)[::2]
        target = torch.arange(count, device=device) % 4 if mode == "pair" else None
        actual = module(x, weight, source, target)
        selected = weight[source, target] if mode == "pair" else weight[source]
        expected = linear(x, selected)
        tolerance = 3e-5 if dtype == torch.float32 else 2e-11
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        for reflected, time_reversal in ((False, False), (True, False), (False, True)):
            torch.testing.assert_close(
                module(
                    _transform(x, linear.irreps_in, 0.37, reflected, time_reversal),
                    weight,
                    source,
                    target,
                ),
                _transform(actual, linear.irreps_out, 0.37, reflected, time_reversal),
                atol=tolerance,
                rtol=tolerance,
            )
        for _ in range(3):
            seed = torch.randn_like(actual)
            gradients = [
                torch.autograd.grad(
                    (y.sin() * seed).sum(), (x, weight), create_graph=True
                )
                for y in (actual, expected)
            ]
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=5 * tolerance, rtol=5 * tolerance)
            actual, expected = [
                torch.cat([g.flatten() for g in gs]) for gs in gradients
            ]


@pytest.mark.parametrize("space,experts", [("o3", 1), ("o3", 2), ("co3", 1), ("o2", 1)])
@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_element_linear_compile(
    space, experts, mode, backend, device, double_precision
):
    from torch.fx.experimental.proxy_tensor import make_fx

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    group = {"o3": eqx_o3, "co3": co3, "o2": o2}[space]
    linear_cls = o3.Linear if space == "o3" else group.Linear
    linear = linear_cls(
        "2x0e+0e+2x1m+2x2m" if space == "o2" else "2x0e+0e+2x1o+2x2e",
        "3x0e+2x1m+2m+3m" if space == "o2" else "3x0e+2x1o+1o+3o",
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    cls = group.ElementLinear if experts == 1 else group.MoEElementLinear
    module = cls(
        linear,
        backend=backend,
        element_mode=mode,
        **({} if experts == 1 else {"num_experts": experts}),
    )
    reference = copy_model(module)
    reference.backend = "torch"
    reference.expand_weights = True
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    shape = (3, 4) if mode == "pair" else (3,)
    weight = torch.randn(
        *shape, experts, linear.weight_numel, device=device, requires_grad=True
    )
    traced = None
    for count in (5, 9, 9, 0):
        features = torch.randn(
            module.input_dim, count, device=device
        ).T.requires_grad_()
        source = torch.randint(3, (count,), device=device)
        target = torch.randint(4, (count,), device=device) if mode == "pair" else None
        args = (features, weight, source, target)
        if traced is None:
            traced = make_fx(module, tracing_mode="symbolic")(*args)
        expected = reference(*args)
        for evaluate in (module, traced, compiled):
            actual = evaluate(*args)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
            for a, b in zip(
                torch.autograd.grad(actual.square().sum(), (features, weight)),
                torch.autograd.grad(
                    expected.square().sum(), (features, weight), retain_graph=True
                ),
            ):
                torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)
