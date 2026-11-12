"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_element_linear_high_degree(backend, double_precision):
    linear = co3.Linear(
        "2x0e+2x2e+2x5o",
        "3x0e+3x2e+3x5o",
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE)
    module = co3.ElementLinear(linear, backend, element_mode="pair")
    x = torch.randn(35, linear.irreps_in.dim, device=DEVICE, requires_grad=True)
    weight = torch.randn(3, 4, linear.weight_numel, device=DEVICE, requires_grad=True)
    source = torch.arange(35, device=DEVICE) % 3
    target = torch.arange(35, device=DEVICE) % 4
    actual, expected = (
        module(x, weight, source, target),
        linear(x, weight[source, target]),
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, weight)),
        torch.autograd.grad(expected.square().sum(), (x, weight)),
    ):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("normalization", ["element", "path"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(
    "instructions", [None, [(1, 0), (0, 0), (3, 2), (2, 1), (0, 0)]]
)
def test_linear(double_precision, normalization, shared, instructions):
    kwargs = dict(
        path_normalization=normalization,
        internal_weights=False,
        shared_weights=shared,
        biases=shared,
        instructions=instructions,
    )
    ref = o3.Linear("2x0e+1x0e+2x1o+1x2e", "3x0e+2x1o+2x2e+1x0o", **kwargs).to(DEVICE)
    module = co3.Linear(
        ref.irreps_in, ref.irreps_out, output_basis="spherical", **kwargs
    ).to(DEVICE)
    x = torch.randn(5, ref.irreps_in.dim, device=DEVICE)
    shape = () if shared else (5,)
    weights = torch.randn(*shape, ref.weight_numel, device=DEVICE)
    bias = torch.randn(*shape, ref.bias_numel, device=DEVICE)
    cx = co3.ChangeOfBasis(ref.irreps_in).to(DEVICE)(x)
    torch.testing.assert_close(
        module(cx, weights, bias), ref(x, weights, bias), atol=2e-14, rtol=2e-14
    )
    for a, b in zip(module.weight_views(weights), ref.weight_views(weights)):
        torch.testing.assert_close(a, b)
    raw = torch.randn_like(cx)
    inverse = co3.ChangeOfBasis(ref.irreps_in, inverse=True).to(DEVICE)
    torch.testing.assert_close(
        module(raw, weights, bias),
        ref(inverse(raw), weights, bias),
        atol=2e-14,
        rtol=2e-14,
    )
    inputs = [x.requires_grad_() for x in (raw, weights, bias)]
    losses = [
        module(*inputs).square().sum(),
        ref(inverse(inputs[0]), inputs[1], inputs[2]).square().sum(),
    ]
    differentiable = [x for x in inputs if x.numel()]
    for _ in range(2):
        gradients = [
            torch.autograd.grad(loss, differentiable, create_graph=True)
            for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
        tangents = [torch.randn_like(x) for x in differentiable]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


def test_linear_disconnected_paths(double_precision):
    for count in (0, 3):
        linear = co3.Linear("0x0e+2x1e", "1x0e+0x1e").to(DEVICE)
        feats = torch.randn(
            count, linear.irreps_in.dim, device=DEVICE, requires_grad=True
        )
        result = linear(feats)
        assert result.shape == (count, 1)
        assert not result.count_nonzero()
        assert not linear.output_mask.count_nonzero()
        assert not torch.autograd.grad(result.sum(), feats)[0].count_nonzero()
