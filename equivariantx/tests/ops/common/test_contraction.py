"""Shared projection scheduling for native convolution programs."""

import pytest
import torch

from equivariantx.ops import config
from equivariantx.ops.common.contraction import project_radial
from equivariantx.utils.metadata import parse_metadata


def test_shared_operands_and_role_alias(monkeypatch):
    monkeypatch.setattr(config, "RADIAL_PROJECT_MIN_WIDTH", 4)
    radial = torch.randn(3, 4, requires_grad=True)
    projection = torch.randn(4, 4, requires_grad=True)
    # Feature and radial roles alias, as do the two frame roles.
    original = repr((((0, 0, 1, 2, 2, 3, 4), False, ((6, 0),)),))
    inputs = [radial, projection, torch.randn(3, 9), torch.randn(3, 1), radial]
    program, values = project_radial(original, inputs, 6)
    mapping, _, _ = parse_metadata(program)[0]
    assert values[mapping[0]] is radial
    assert mapping[3] == mapping[4]
    assert mapping[0] != mapping[1]
    torch.testing.assert_close(values[mapping[1]], radial @ projection)
    torch.testing.assert_close(inputs[0], radial)
    with torch.no_grad():
        program, values = project_radial(original, inputs, 6)
    assert program == original and values is inputs


@pytest.mark.parametrize("bounded", [False, True])
def test_symbolic_projection_bound(bounded, monkeypatch):
    monkeypatch.setattr(config, "RADIAL_PROJECT_MAX_BYTES", 16 * 32 * 4)

    class Projection(torch.nn.Module):
        def forward(self, radial, matrix):
            program = repr((((0, 1, 2, 3, 4), False, ((4, 0),)),))
            program, values = project_radial(
                program, [radial, radial, matrix, radial, radial], 4
            )
            return values[parse_metadata(program)[0][0][1]]

    rows = torch.export.Dim("rows", min=2, max=16 if bounded else None)
    inputs = (torch.randn(7, 1024), torch.randn(1024, 32))
    exported = torch.export.export(
        Projection(), inputs, dynamic_shapes=({0: rows}, {}), strict=True
    )
    assert sum(
        node.target == torch.ops.aten.matmul.default for node in exported.graph.nodes
    ) == int(bounded)
    for count in (7, 11):
        radial = torch.randn(count, 1024)
        expected = radial @ inputs[1] if bounded else radial
        torch.testing.assert_close(exported.module()(radial, inputs[1]), expected)
