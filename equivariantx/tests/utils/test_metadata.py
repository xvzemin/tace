"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.utils.metadata import static_dimensions


def test_dimension_specialization():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def function(value):
        width = static_dimensions(value, (-1,))[0]
        assert repr(width) in ("3", "17")
        return value + width

    compiled = torch.compile(function, backend=backend, fullgraph=True, dynamic=True)
    for rows, width in ((7, 3), (11, 3), (7, 17), (15, 17)):
        value = torch.randn(rows, width)
        torch.testing.assert_close(compiled(value), value + width)
    assert len(graphs) == 2


def test_shared_metadata_cache():
    from equivariantx.utils.metadata import compact_metadata, parse_metadata

    specification = (2, ((0, 1, "uvu", True, 0.5),))
    metadata = repr(specification)
    result = parse_metadata(metadata)
    assert result == specification
    assert parse_metadata(metadata) is result
    assert compact_metadata(metadata) == metadata
    large = repr(specification * 1000)
    packed = compact_metadata(large)
    assert len(packed) < len(large)
    assert compact_metadata(packed) == packed
    assert parse_metadata(packed) == specification * 1000
    assert parse_metadata(packed) is parse_metadata(packed)
    assert parse_metadata("None") is None
    with pytest.raises(ValueError):
        parse_metadata("tuple()")


def test_dimension_specialization_inside_nested_transforms():
    def loss(value):
        width = static_dimensions(value, (-1,))[0]
        assert repr(width) == "3"
        return (value * width).sin().sum()

    def function(value, tangent):
        return torch.func.jvp(torch.func.grad(loss), (value,), (tangent,))

    compiled = torch.compile(
        function, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for rows in (7, 11):
        value, tangent = torch.randn(rows, 3), torch.randn(rows, 3)
        torch.testing.assert_close(compiled(value, tangent), function(value, tangent))


def test_compressed_plans_compilation():
    from equivariantx.utils.metadata import decode_plans, encode_plans

    def function(value, specification):
        scale, offset = decode_plans(specification)[0]
        return value * scale + offset

    decode_plans.cache_clear()
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    value = torch.randn(3, 5)
    for plan in ((2, 3), (4, -1)):
        specification = encode_plans([repr(plan)])
        torch.testing.assert_close(
            compiled(value, specification), value * plan[0] + plan[1]
        )
