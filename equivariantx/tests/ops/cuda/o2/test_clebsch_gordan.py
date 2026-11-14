"""Native O(2) force-loss packages load without EQX in a separate process."""

import pytest
import torch

from equivariantx.tests.o2.testing import check_joint_o2_aoti


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native O(2) requires CUDA")
def test_joint_training_package(tmp_path):
    check_joint_o2_aoti("clebsch_gordan", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native O(2) requires CUDA")
def test_dynamic_inductor():
    from equivariantx import o2
    from equivariantx.o2.clebsch_gordan import clebsch_gordan_product
    from equivariantx.ops.torch.o2.clebsch_gordan import reference

    a, b, out = o2.Irrep("1m"), o2.Irrep("2m"), o2.Irrep("1m")
    weight = torch.randn(1, 2, 4, device="cuda", dtype=torch.float64)

    def operation(x):
        return clebsch_gordan_product(
            x, a, weight, b, out, elementwise=True, backend="cuda"
        )

    def ordinary(x):
        return reference(x, a, weight, b, out, elementwise=True)

    shape = (2, 4)
    compiled = torch.compile(operation, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, *shape, device="cuda", dtype=torch.float64) * 0.1 + 0.5
        torch.testing.assert_close(compiled(x), ordinary(x), atol=3e-10, rtol=3e-10)
