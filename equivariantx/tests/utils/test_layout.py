"""Feature permutations composed with nonlinear upstream operations."""

import pytest
import torch

from equivariantx.ops import config
from equivariantx.tests.ops.testing import assert_functional_derivatives
from equivariantx.utils.layout import permute


@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(compiler):
    values = torch.randn(2, 3, dtype=torch.float64)
    index = torch.tensor([2, 0, 1])

    def function(value, index):
        return permute(value.sin(), index, index.argsort())

    def reference(value, index):
        return value.sin().index_select(-1, index)

    assert_functional_derivatives(function, reference, (values, index), compiler)


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("rows", [0, 4])
def test_reverse_derivatives(reference, rows, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    values = torch.randn(rows, 3, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([2, 0, 1])
    actual = permute(values, index, index.argsort())
    expected = values.index_select(-1, index)
    for _ in range(4):
        torch.testing.assert_close(actual, expected)
        actual, expected = [
            torch.autograd.grad(value.sin().sum(), values, create_graph=True)[0]
            for value in (actual, expected)
        ]
