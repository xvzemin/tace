"""Native quaternion programs preserve ordinary forward-mode tangents."""

import pytest
import torch

from equivariantx.ops.common.quaternion import quaternion_polynomial
from equivariantx.ops.common.torch.quaternion import quaternion_polynomial as reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA quaternion polynomials")
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("directions", [0, 1, 3])
def test_ordinary_forward_ad(transpose, directions, double_precision):
    values = tuple(
        torch.randn(3, width, device="cuda") * 0.2
        for width in ([4] * (directions + 1) + ([35] if transpose else []))
    )
    tangents = tuple(torch.randn_like(value) for value in values)
    with torch.autograd.forward_ad.dual_level():
        duals = [
            torch.autograd.forward_ad.make_dual(value, tangent)
            for value, tangent in zip(values, tangents)
        ]
        result = torch.autograd.forward_ad.unpack_dual(
            quaternion_polynomial(2, transpose, duals)
        )
    expected = torch.autograd.functional.jvp(
        lambda *args: reference(2, transpose, args), values, tangents
    )
    assert result.tangent is not None
    torch.testing.assert_close(tuple(result), expected, atol=2e-10, rtol=2e-10)
