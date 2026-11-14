"""Functional derivatives of native semi-coupled contractions."""

import pytest
import torch
from e3nn import o3

from equivariantx.mbe.ace import SemiCoupledSymmetricContraction
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA ACE")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(compiler, double_precision):
    irreps = "2x0e+2x1o"
    product = o3.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", False), (1, 1, 0, "uuu", False), (0, 1, 1, "uuu", False)],
        internal_weights=False,
        shared_weights=False,
    )
    linears = [
        o3.Linear(irreps, irreps, internal_weights=False, shared_weights=False)
        for _ in range(3)
    ]
    module, reference = [
        SemiCoupledSymmetricContraction([product, product], linears, backend).cuda()
        for backend in ("cuda", "torch")
    ]
    values = (
        torch.randn(5, module.input_dim, device="cuda") * 0.1,
        *(torch.randn(3, linear.weight_numel, device="cuda") for linear in linears),
        torch.tensor([0, 1, 1, 2, 0], device="cuda"),
    )
    assert_functional_derivatives(
        lambda x, *args: module(x, args[:-1], args[-1]),
        lambda x, *args: reference(x, args[:-1], args[-1]),
        values,
        compiler,
    )
