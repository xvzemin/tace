"""Tests for the corresponding EQX source module."""

import pytest
import torch


@pytest.mark.parametrize("heads", [1, 3, 8])
@pytest.mark.parametrize("weighted", [False, True])
def test_graph_reduction(heads, weighted, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.cuda.reduction import reduce_messages

    target = torch.randint(5, (73,), device="cuda")
    channels = heads * 5
    values = torch.randn(73, channels * 3, device="cuda")
    weights = torch.rand(73, heads, device="cuda") if weighted else None
    actual = torch.randn(7, values.shape[1], device="cuda")
    message = (
        values
        if weights is None
        else (values.reshape(73, 3, heads, 5) * weights[:, None, :, None]).flatten(1)
    )
    expected = actual.clone().index_add_(0, target, message)
    reduce_messages(values, target, actual, weights, channels)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
