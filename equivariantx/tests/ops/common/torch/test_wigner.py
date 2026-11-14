"""Native geometry programs and their ordinary Torch evaluator."""

import pytest
import torch

from equivariantx.ops import get_kernel
from equivariantx.utils.scalar import alignment_program


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA geometry reference")
@pytest.mark.parametrize(
    "key", [None, "normalize", "direction_gradient", "quaternion", "wigner1"]
)
def test_alignment_program(key, double_precision):
    nodes, outputs, widths = alignment_program(key, "double")
    values = [torch.randn(5, width, device="cuda") + 0.4 for width in widths]
    for order in range(3):
        result = get_kernel("alignment_cuda", "torch")(repr(key), values)
        expected = get_kernel("alignment_cuda", "cuda")(repr(key), values)
        torch.testing.assert_close(result, expected, atol=2e-9, rtol=2e-9)
        _, outputs, widths = alignment_program(key, "double")
        values += [torch.randn(5, len(group), device="cuda") / 3 for group in outputs]
        key = key, tuple(range(len(widths)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Wigner contraction")
@pytest.mark.parametrize("role", [0, 1, 2])
def test_rotation_transposes(role, double_precision):
    from e3nn import o3

    cg = o3.wigner_3j(1, 2, 3, device="cuda")
    values = [torch.randn(5, width, device="cuda") for width in (9, 25, 49)]
    actual = get_kernel("rotation", "torch")(cg, role, values)
    expected = get_kernel("rotation", "cuda")(cg, role, values)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
