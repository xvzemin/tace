"""Group packing and sparse sums against explicit dense maps."""

import pytest
import torch

from equivariantx.ops import get_kernel


def column_map(device):
    values = (
        [0, 2, 1, 3, 0, 3, 4],
        [0, 2, 4, 7],
        [0, 2, 1, 0, 1, 2, 2],
        [0, 2, 3, 4, 6, 7],
        [2, 2, 3],
    )
    return tuple(
        torch.tensor(value, device=device, dtype=torch.long) for value in values
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [0, 5])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_indexed_sum(device, rows, transpose, normalized, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    index, ptr, ti, tp, count = column_map(device)
    dense = torch.tensor(
        [[1.0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 1, 1]], device=device
    )
    if normalized:
        dense = dense / count[:, None]
    else:
        count = count[:0]
    if transpose:
        index, ptr, ti, tp = ti, tp, index, ptr
    features = torch.randn(rows, 6 if transpose else 10, device=device)[
        :, ::2
    ].requires_grad_()
    arguments = features, index, ptr, ti, tp, count, transpose
    actual = get_kernel("indexed_sum", "torch")(*arguments)
    expected = features @ (dense if transpose else dense.T)
    torch.testing.assert_close(actual, expected)
    if device == "cuda":
        torch.testing.assert_close(
            get_kernel("indexed_sum", "cuda")(*arguments), expected
        )
    torch.testing.assert_close(
        torch.autograd.grad(actual.sin().sum(), features),
        torch.autograd.grad(expected.sin().sum(), features),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [0, 5])
def test_grouped_permutation(device, rows, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(rows, 10, device=device)[:, ::2].requires_grad_()
    indices = torch.tensor([3, 0, 4, 2, 1], device=device)
    inverse = indices.argsort()
    widths = repr((2, 0, 3))
    function = get_kernel("grouped_permute", "torch")
    actual = function(x, indices, inverse, widths)
    expected = torch.cat((x[:, [3, 0]].flatten(), x[:, [4, 2, 1]].flatten())).reshape(
        rows, 5
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        function(actual, inverse, indices, widths, True), x, atol=0, rtol=0
    )
    if device == "cuda":
        torch.testing.assert_close(
            get_kernel("grouped_permute", "cuda")(x, indices, inverse, widths),
            expected,
            atol=0,
            rtol=0,
        )
