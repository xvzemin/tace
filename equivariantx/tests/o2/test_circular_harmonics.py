"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
from equivariantx.tests.o2.testing import _transform
from equivariantx.tests.o2.testing import _time_reverse


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reflected", [False, True])
def test_circular_harmonics_is_equivariant(normalize, reflected):
    module = o2.CircularHarmonics(4, normalize=normalize).to(DEVICE, DTYPE)
    vectors = torch.randn(9, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    transformed = _transform(vectors, module.irreps_in, angle, reflected)
    expected = _transform(module(vectors), module.irreps_out, angle, reflected)
    torch.testing.assert_close(module(transformed), expected)


def test_time_odd_circular_harmonics_alternate_time_parity():
    harmonics = o2.CircularHarmonics(4, time_reversal=True).to(DEVICE, DTYPE)
    assert harmonics.irreps_in == o2.Irreps("1mo")
    assert harmonics.irreps_out == o2.Irreps("0ee+1mo+2me+3mo+4me")
    vectors = torch.randn(8, 2, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        harmonics(-vectors),
        _time_reverse(harmonics(vectors), harmonics.irreps_out),
    )
