"""Tests for the corresponding convolution module."""

import pytest

from equivariantx.o3 import GauntTensorProduct
from equivariantx.tests.conv.testing import check_product_and_convolution_compile


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_product_and_convolution_compile(device, double_precision):
    check_product_and_convolution_compile(GauntTensorProduct, device, double_precision)
