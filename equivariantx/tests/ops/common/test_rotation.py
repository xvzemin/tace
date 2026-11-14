"""Degree-wise frame rotation entrypoints."""

import pytest
import torch

from equivariantx.ops.common import rotation
from equivariantx.tests.utils.testing import check_method_precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frame rotation")
def test_public_method_precision(double_precision):
    features = torch.randn(3, 4, 2, device="cuda")
    matrix = torch.randn(3, 4, 4, device="cuda")
    metadata = repr(((0, 1, 1, 1), (0, 1, 1, 1)))
    check_method_precision(rotation, [("rotate", (features, matrix, metadata), {})])
