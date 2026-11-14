"""Native O(2) force-loss packages load without EQX in a separate process."""

import pytest
import torch

from equivariantx.ops import config
from equivariantx.tests.o2.testing import check_joint_o2_aoti


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native O(2) requires CUDA")
@pytest.mark.parametrize("schedule", ["default", "native"])
def test_joint_training_package(tmp_path, monkeypatch, schedule):
    if schedule == "native":
        monkeypatch.setattr(config, "O2_CONTRACTION_STAGED", True)
        monkeypatch.setattr(config, "O2_CONTRACTION_INLINE", False)
    check_joint_o2_aoti("contraction", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native O(2) requires CUDA")
@pytest.mark.parametrize("schedule", ["default", "native"])
def test_dynamic_inductor(monkeypatch, schedule):
    if schedule == "native":
        monkeypatch.setattr(config, "O2_CONTRACTION_STAGED", True)
        monkeypatch.setattr(config, "O2_CONTRACTION_INLINE", False)
    from equivariantx import o2

    args = "2x0e+2x1m", "2x0e+2x1m", 3
    layer = o2.SymmetricContraction(*args, backend="cuda").cuda().double()
    reference = o2.SymmetricContraction(*args, algorithm="dense").cuda().double()
    weight = torch.randn(layer.weight_numel, device="cuda", dtype=torch.float64) * 0.1

    def operation(x):
        return layer(x, weight)

    def ordinary(x):
        return reference(x, weight)

    shape = (layer.irreps_in.dim,)
    compiled = torch.compile(operation, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, *shape, device="cuda", dtype=torch.float64) * 0.1 + 0.5
        torch.testing.assert_close(compiled(x), ordinary(x), atol=3e-10, rtol=3e-10)
