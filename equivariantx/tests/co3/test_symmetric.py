"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx.co3.symmetric import SymmetricBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_symmetric_basis_adjoint(double_precision, rank):
    basis = SymmetricBasis(rank).to(DEVICE)
    tensor = torch.randn(2, 3**rank, device=DEVICE, requires_grad=True)
    compact = torch.randn(2, basis.dim, device=DEVICE, requires_grad=True)
    projected = basis.pack(tensor)
    loss = (projected * compact).sum()
    expected = (tensor * basis.unpack(compact)).sum()
    torch.testing.assert_close(loss, expected, atol=3e-13, rtol=3e-13)
    grad = torch.autograd.grad(loss, tensor, create_graph=True)[0]
    torch.testing.assert_close(grad, basis.unpack(compact), atol=3e-13, rtol=3e-13)
