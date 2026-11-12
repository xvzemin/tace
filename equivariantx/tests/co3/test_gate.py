"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import nn
from equivariantx import co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_gate_and_elementwise(double_precision):
    args = (
        "2x0e+1x0o",
        [torch.nn.functional.silu, torch.tanh],
        "2x0e+2x0o",
        [torch.sigmoid, torch.tanh],
        "2x1o+1x2e+1x2o",
    )
    ref = nn.Gate(*args).to(DEVICE)
    module = co3.Gate(*args).to(DEVICE)
    assert module.irreps_in.spherical() == ref.irreps_in
    assert module.irreps_out.spherical() == ref.irreps_out
    x = torch.randn(8, ref.irreps_in.dim, device=DEVICE, requires_grad=True)
    actual = co3.ChangeOfBasis(module.irreps_out, True).to(DEVICE)(
        module(co3.ChangeOfBasis(ref.irreps_in).to(DEVICE)(x))
    )
    torch.testing.assert_close(actual, ref(x), atol=3e-14, rtol=3e-14)
    with pytest.raises(ValueError):
        co3.Gate("", [], "1x0o", [torch.sigmoid], "1x1e")
