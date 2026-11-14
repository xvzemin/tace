"""Exact normalized symmetric-power matrix plans."""

import pytest
import torch

from equivariantx.co3.symmetric import SymmetricBasis
from equivariantx.ops.common.symmetric import matrix_inputs, matrix_metadata
from equivariantx.ops.common.torch.symmetric import evaluate_torch
from equivariantx.ops.torch.co3.symmetric import transform_reference


@pytest.mark.parametrize("rank", [0, 1, 2, 4, 6])
def test_representation(rank):
    basis = SymmetricBasis(rank).double()
    matrix = (torch.randn(2, 3, 3, dtype=torch.float64) * 0.2).requires_grad_()
    program, types, operands = matrix_inputs(matrix, rank)
    actual = evaluate_torch(matrix_metadata(rank), program, types, operands)[0]
    actual = actual.reshape(2, basis.dim, basis.dim)
    expected = transform_reference(
        basis,
        torch.eye(basis.dim, dtype=torch.float64).expand(2, -1, -1),
        matrix.unsqueeze(1),
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    if rank:
        for _ in range(3):
            actual, expected = [
                torch.autograd.grad(value.sin().sum(), matrix, create_graph=True)[0]
                for value in (actual, expected)
            ]
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
