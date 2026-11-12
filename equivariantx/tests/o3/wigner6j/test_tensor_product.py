"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o3 as eqx_o3

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize(
    "irrep_normalization,path_normalization",
    [("component", "element"), ("norm", "path"), ("none", "none")],
)
def test_wigner6j_tensor_product(
    backend, irrep_normalization, path_normalization, double_precision
):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = eqx_o3.Wigner6jTensorProduct(
        "2x1o",
        "3x1o",
        "2x1e",
        "2x0e+2x1e+2x2e",
        "2x0e+2x1e+2x2e",
        [(0, 0, k, "uvu", k != 1) for k in range(3)] + [(0, 0, 0, "uvu", True, 0.5)],
        [
            (i, 0, k, "uvu", i != 2)
            for i in range(3)
            for k in range(3)
            if abs(i - 1) <= k <= i + 1
        ],
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(device)
    inputs = [
        torch.randn(4, width, device=device, requires_grad=True)
        for width in (6, 9, 6, module.weight_numel1, module.weight_numel2)
    ]
    actual, expected = module(*inputs), module.forward_reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((value.sin() * seed).sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]


@pytest.mark.parametrize("degrees", [(0, 2, 1), (2, 3, 2), (4, 3, 2)])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_wigner6j_shared_weights(degrees, backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    l1, l2, l3 = degrees
    ir1, ir2, ir3 = o3.Irrep(l1, -1), o3.Irrep(l2, 1), o3.Irrep(l3, -1)
    mid = o3.Irreps([(2, ir) for ir in ir1 * ir2])
    out = o3.Irreps([(2, o3.Irrep(l, 1)) for l in range(l1 + l2 + l3 + 1)])
    module = eqx_o3.Wigner6jTensorProduct(
        [(2, ir1)],
        [(1, ir2)],
        [(1, ir3)],
        mid,
        out,
        [(0, 0, i, "uvu", True) for i in range(len(mid))],
        [
            (i, 0, k, "uvu", True)
            for i, (_, ir) in enumerate(mid)
            for k, (_, ir_out) in enumerate(out)
            if ir_out in ir * ir3
        ],
        backend=backend,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)
    x1 = torch.randn(2, 1, 2 * ir1.dim, device=device)
    x2 = torch.randn(3, ir2.dim, device=device)
    x3 = torch.randn(ir3.dim, device=device)
    actual, expected = module(x1, x2, x3), module.forward_reference(x1, x2, x3)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), tuple(module.parameters())),
        torch.autograd.grad(expected.square().sum(), tuple(module.parameters())),
    ):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
