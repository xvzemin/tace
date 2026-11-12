"""Functional derivatives of native indexed linear maps."""

import pytest
import torch
from e3nn import o3

from equivariantx import o2
from equivariantx import o3 as eqx_o3
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA maps")
@pytest.mark.parametrize("space,experts", [("o2", 1), ("o3", 1), ("o3", 2)])
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(space, experts, compiler, double_precision):
    namespace, irreps = (o2, "2x0e+3x1m") if space == "o2" else (o3, "2x0e+3x1o")
    public = o2 if space == "o2" else eqx_o3
    linear = namespace.Linear(
        irreps, irreps, internal_weights=False, shared_weights=False
    )
    constructor = public.ElementLinear if space == "o2" else public.MoEElementLinear
    options = {} if space == "o2" else {"num_experts": experts}
    module, reference = [
        constructor(linear, backend=backend, element_mode="single", **options).cuda()
        for backend in ("cuda", "torch")
    ]
    values = (
        torch.randn(5, module.input_dim, device="cuda"),
        torch.randn(3, experts, module.weight_numel, device="cuda"),
        torch.tensor([0, 1, 1, 2, 0], device="cuda"),
    )
    assert_functional_derivatives(module, reference, values, compiler)
