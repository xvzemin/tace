"""Tests for the corresponding EQX source module."""

import math
import pytest
import torch
from e3nn import o3
from equivariantx import o3 as eqx_o3

IRREPS = "0e+0o+1e+1o+2e+2o"


def test_matrix_matches_cg_reference(double_precision):
    module = eqx_o3.MatrixTensorProduct(
        IRREPS, IRREPS, IRREPS, irrep_normalization="none"
    )
    basis = {l: math.sqrt(2 * l + 1) * o3.wigner_3j(1, 1, l) for l in range(3)}
    instructions, weights = [], []
    for i, (_, ir1) in enumerate(module.irreps_in1):
        for j, (_, ir2) in enumerate(module.irreps_in2):
            for k, (_, ir) in enumerate(module.irreps_out):
                if ir not in ir1 * ir2:
                    continue
                coefficient = torch.einsum(
                    "abi,bcj,ack->ijk", basis[ir1.l], basis[ir2.l], basis[ir.l]
                )
                weights.append((coefficient * o3.wigner_3j(ir1.l, ir2.l, ir.l)).sum())
                instructions.append((i, j, k, "uuu", True))
    reference = o3.TensorProduct(
        IRREPS,
        IRREPS,
        IRREPS,
        instructions,
        internal_weights=False,
        irrep_normalization="none",
        path_normalization="none",
    )
    x, y = torch.randn(2, 4, module.irreps_in1.dim)
    torch.testing.assert_close(
        module(x, y, torch.ones(module.weight_numel)),
        reference(x, y, torch.stack(weights)),
        atol=2e-13,
        rtol=2e-13,
    )


def test_matrix_degree_and_structural_zero(double_precision):
    module = eqx_o3.MatrixTensorProduct("3o", "5o", "5e", matrix_degree=3)
    assert not module.output_mask.count_nonzero()
    assert not module(torch.randn(2, 7), torch.randn(2, 11)).count_nonzero()
    assert eqx_o3.MatrixTensorProduct("3o", "0e", "3o").matrix_degree == 2
    with pytest.raises(ValueError, match="matrix_degree"):
        eqx_o3.MatrixTensorProduct("3o", "0e", "3o", matrix_degree=1)
