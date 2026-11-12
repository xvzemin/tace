"""Functional derivatives of fused product, gates and expert maps."""

import pytest
import torch
from e3nn import o3

from equivariantx.mbe.ace import BilinearContraction
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(compiler, double_precision):
    irreps = "4x0e+4x1o"
    product = o3.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", True), (1, 1, 0, "uuu", True), (0, 1, 1, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = o3.Linear(
        "2x0e+2x1o", "2x0e+2x1o", internal_weights=False, shared_weights=False
    )
    shared = o3.Linear(irreps, irreps, internal_weights=False, shared_weights=False)
    module, reference = [
        BilinearContraction(product, linear, 2, backend, shared).cuda()
        for backend in ("cuda", "torch")
    ]
    values = (
        torch.randn(5, module.input_dim, device="cuda"),
        torch.randn(5, module.base_dim, device="cuda"),
        torch.randn(5, module.gate_dim, device="cuda"),
        torch.randn(3, 2, linear.weight_numel, device="cuda"),
        torch.randn(shared.weight_numel, device="cuda"),
        torch.tensor([0, 1, 1, 2, 0], device="cuda"),
    )
    assert_functional_derivatives(
        lambda x, y, g, w, s, t: module(x, y, g, w, t, s),
        lambda x, y, g, w, s, t: reference(x, y, g, w, t, s),
        values,
        compiler,
    )
