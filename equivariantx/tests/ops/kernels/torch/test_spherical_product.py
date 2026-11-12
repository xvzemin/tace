"""Grouped CG paths and exact coefficient dtype conversions."""

import pytest
import torch
from e3nn import o3

from equivariantx.ops.kernels.torch.spherical_product import SphericalProduct


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_repeated_paths(backend, device, double_precision):
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA unavailable")
    irreps = tuple(o3.Irreps("2x1o+2x0e+2x1o") for _ in range(3))
    instructions = [
        (i, j, k, "uvu", True, float(1 + i + j + k))
        for i, (_, a) in enumerate(irreps[0])
        for j, (_, b) in enumerate(irreps[1])
        for k, (_, c) in enumerate(irreps[2])
        if c in a * b
    ]
    reference = o3.TensorProduct(*irreps, instructions, shared_weights=False).to(device)
    module = (
        SphericalProduct(irreps, reference.instructions, backend)
        .to(device)
        .float()
        .double()
    )
    for rows in (4, 0):
        values = tuple(
            torch.randn(*shape, device=device, requires_grad=True)
            for shape in (
                (rows, irreps[0].dim),
                (1, irreps[1].dim),
                (1, reference.weight_numel),
            )
        )
        actual, expected = module(*values), reference(*values)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        ga, gb = [
            torch.autograd.grad(value.square().sum(), values)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-11, rtol=2e-11)
