"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import conv as eqx_conv


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("output_basis", ["spherical", "cartesian"])
def test_cartesian_compact_linear(backend, output_basis, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3
    from equivariantx.conv.co3.linear import Linear

    tp = co3.TensorProduct(
        "2x0e+2x1o+2x2e",
        "1o+2e",
        "2x2e+2x2e+2x1e+2x3o+2x2o+2x1o",
        [
            (0, 1, 0, "uvu", True),
            (1, 0, 1, "uvu", True),
            (1, 0, 2, "uvu", True),
            (2, 0, 3, "uvu", True),
            (2, 0, 4, "uvu", True),
            (2, 0, 5, "uvu", True),
        ],
        project=False,
        internal_weights=False,
        shared_weights=False,
    )
    compact = eqx_conv.CartesianTensorProductConv(
        tp,
        backend=backend,
        normalize=False,
        symmetric_inputs=True,
        input_basis="spherical",
        compact_output=True,
    )
    linear = Linear(
        tp.irreps_out.simplify(),
        "3x1o+3x1e+3x2e+3x2o+3x3o",
        compact.harmonic_output_index,
        internal_weights=False,
        backend=backend,
        output_basis=output_basis,
    )
    # Reconstruct constants when changing model precision.
    compact, linear = [m.float().cuda().double() for m in (compact, linear)]
    reference = eqx_conv.CartesianTensorProductConv(
        tp, backend="torch", normalize=False
    ).cuda()
    basis = co3.ChangeOfBasis(tp.irreps_in1).cuda()
    reference_linear = co3.Linear(
        linear.irreps_in,
        linear.irreps_out,
        internal_weights=False,
        output_basis=output_basis,
        project=output_basis == "cartesian",
    ).cuda()
    edges = torch.randint(4, (2, 11), device="cuda")
    inputs = [
        torch.randn(4, tp.irreps_in1.spherical().dim, device="cuda"),
        torch.randn(11, 3, device="cuda") * 0.2,
        torch.randn(11, 2, device="cuda"),
        torch.randn(2, tp.weight_numel, device="cuda"),
        torch.randn(linear.weight_numel, device="cuda"),
    ]
    inputs[1][0] = 0
    inputs = [x.requires_grad_() for x in inputs]
    x, vectors, radial, projection, weight = inputs
    full = reference(basis(x), None, radial, projection, edges, vectors=vectors)
    packed = compact(x, None, radial, projection, edges, vectors=vectors)
    assert packed.shape[-1] < full.shape[-1]
    torch.testing.assert_close(
        packed[:, compact.harmonic_output_index], full, atol=2e-12, rtol=2e-12
    )
    outputs = [linear(packed, weight), reference_linear(full, weight)]
    torch.testing.assert_close(*outputs, atol=2e-12, rtol=2e-12)
    losses = [out.sin().sum() for out in outputs]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        tangents = [torch.randn_like(x) for x in inputs]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
