"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx.mbe.ace import (
    IctdBilinearContraction,
    IctdSemiCoupledSymmetricContraction,
)


@pytest.mark.parametrize(
    "kind",
    [
        "ace",
        "ace_split",
        "bilinear",
        "bilinear_split",
        "bilinear_wide",
        "moe",
        "moe_split",
    ],
)
def test_ictd_contractions(kind, double_precision, monkeypatch):
    from equivariantx import co3

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if kind == "ace_split":
        from equivariantx.ops.kernels.cuda import ace

        batches = ace.schedule_batches
        monkeypatch.setattr(ace, "schedule_batches", lambda groups: batches(groups, 7))
        monkeypatch.setattr(ace, "weighted_source", lambda *args: None)
    elif kind.endswith("_split"):
        from equivariantx.ops.kernels.cuda import bilinear

        batches = bilinear.metadata_batches
        monkeypatch.setattr(
            bilinear, "metadata_batches", lambda value: batches(value, 1)
        )
    channels = 256 if kind == "bilinear_wide" else 4
    irreps = co3.Irreps(f"{channels}x0e+{channels}x1o+{channels}x2e")
    output = co3.Irreps(f"{channels}x0e+{channels}x1e+{channels}x2e+{channels}x3o")
    weighted = not kind.startswith("ace")
    tp = co3.TensorProduct(
        irreps,
        irreps,
        output,
        [
            (1, 1, 0, "uuu", weighted),
            (1, 1, 1, "uuu", weighted),
            (1, 1, 2, "uuu", weighted),
            (1, 2, 3, "uuu", weighted),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    experts = 2 if kind.startswith("moe") else 1
    coefficient = co3.Linear(
        [(mul // experts, ir) for mul, ir in output],
        [(mul // experts, ir) for mul, ir in output],
        internal_weights=False,
        shared_weights=False,
    )
    if weighted:
        modules = [
            IctdBilinearContraction(tp, coefficient, experts, backend=backend)
            for backend in ("torch", "cuda")
        ]
    else:
        first = co3.Linear(irreps, output, internal_weights=False, shared_weights=False)
        modules = [
            IctdSemiCoupledSymmetricContraction(
                [tp], [first, coefficient], backend=backend
            )
            for backend in ("torch", "cuda")
        ]
    modules = [module.cuda() for module in modules]
    transform = co3.ChangeOfBasis(irreps).cuda()
    source = torch.randn(5, irreps.spherical().dim, device="cuda", requires_grad=True)
    features = transform(source)
    types = torch.tensor([0, 1, 0, 2, 1], device="cuda")
    if weighted:
        gates = torch.randn(5, tp.weight_numel, device="cuda", requires_grad=True)
        weight = torch.randn(
            3, experts, coefficient.weight_numel, device="cuda", requires_grad=True
        )
        inputs = (source, gates, weight)
        outputs = [
            module(features, features, gates, weight, types) for module in modules
        ]
    else:
        weights = [
            torch.randn(3, linear.weight_numel, device="cuda", requires_grad=True)
            for linear in (first, coefficient)
        ]
        inputs = (source, *weights)
        outputs = [module(features, weights, types) for module in modules]
    torch.testing.assert_close(*outputs, atol=1e-11, rtol=1e-11)
    losses = [output.sin().sum() for output in outputs]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for expected, actual in zip(*gradients):
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        tangents = [torch.randn_like(value) for value in inputs]
        losses = [
            sum((value * tangent).sum() for value, tangent in zip(grad, tangents))
            for grad in gradients
        ]
