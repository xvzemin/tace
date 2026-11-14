"""Native complete Cartesian gradients in independently loaded AOTI packages."""

import pytest
import torch

from equivariantx.ops import config
from equivariantx.tests.utils.testing import SPACES, check_ictd_native_aoti

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native sparse chains require CUDA"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_aoti(dtype, monkeypatch, tmp_path):
    check_ictd_native_aoti(2, dtype, monkeypatch, tmp_path)


def test_native_dynamic(monkeypatch):
    monkeypatch.setattr(config, "ICTD_INLINE", False)
    module = SPACES[2].ICTD(3, backend="cuda", dtype=torch.float64, device="cuda")
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    matrix = module.change_of_basis
    for count in (3, 7, 0):
        x = torch.randn(count, 2, 8, device="cuda", dtype=torch.float64) * 0.1
        torch.testing.assert_close(compiled(x), x @ matrix, atol=3e-11, rtol=3e-11)
