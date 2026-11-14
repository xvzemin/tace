"""Exact harmonic coefficients and their expression programs."""

import pytest
import torch
from e3nn import o3

from equivariantx.ops.common.torch.edge import evaluate_torch
from equivariantx.utils.harmonics import harmonic_program
from equivariantx.utils.program import Program


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 7])
@pytest.mark.parametrize("normalization", ["norm", "component", "integral"])
def test_harmonic_program(degree, normalization, double_precision):
    vectors = torch.cat(
        (torch.eye(3), -torch.eye(3), torch.zeros(1, 3), torch.randn(2, 3) / 2)
    )
    vectors.requires_grad_()
    program = Program()
    output = harmonic_program(
        program, program.input(0, "edge", 3), degree, normalization
    )
    metadata = repr((tuple(program.nodes), ((output, 0, "edge"),)))
    index = torch.arange(len(vectors))
    actual = evaluate_torch(metadata, [vectors], index, index, len(vectors))[0]
    expected = o3.spherical_harmonics(
        degree, vectors, normalize=False, normalization=normalization
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    if not degree:
        return
    direction = torch.randn_like(vectors) / 4
    a, b = actual.sin().sum(), expected.sin().sum()
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value, vectors, create_graph=True)[0]
            for value in (a, b)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-9, rtol=2e-9)
        a, b = (ga * direction).sum(), (gb * direction).sum()
