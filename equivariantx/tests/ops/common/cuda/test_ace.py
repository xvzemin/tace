"""Tests for the corresponding EQX source module."""

import pytest
import torch


def test_single_writer_covers_interleaved_outputs():
    from equivariantx.ops.common.cuda.ace import single_writer

    even = (0, 3, 2, 1)
    odd = (1, 3, 2, 1)
    assert single_writer([even, odd], 6)
    assert single_writer([(0, 2, 3, 3)], 6)
    assert not single_writer([even], 6)
    assert single_writer([even], 6, complete=False)
    assert not single_writer([even, odd, even], 6, complete=False)
    assert not single_writer([even, odd], 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA provider")
def test_partial_channel_overlap():
    from equivariantx.o3.symmetric_contraction import make_metadata
    from equivariantx.ops.common.ace import ace_inputs, contract

    metadata = make_metadata(
        [((s, s, s, 0, 3, 0, 1, 1, 1), 1.0) for s in (0, 2)], (5, 5, 5), False
    )
    x, y = (
        torch.randn(21, 5, device="cuda", dtype=torch.float64, requires_grad=True)
        for _ in range(2)
    )
    types = torch.zeros(21, device="cuda", dtype=torch.long)
    program, operands = ace_inputs(x, y, None, 5)
    (out,) = contract(metadata, program, types, operands)
    expected = x * y * x.new_tensor([1, 1, 2, 1, 1])
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        torch.autograd.grad(out.sum(), (x, y)),
        torch.autograd.grad(expected.sum(), (x, y)),
    )


def test_contraction_source_reuses_polynomials():
    from equivariantx.ops.common.cuda.codegen import contraction_source

    cache, lines = {}, []
    value = contraction_source([(2.0, ("x", "y")), (3.0, ("z", "x"))], cache, lines)
    count = len(lines)
    assert (
        contraction_source(
            [(3.0, ("x", "z")), (1.0, ("y", "x")), (1.0, ("x", "y"))],
            cache,
            lines,
        )
        == value
    )
    assert (
        contraction_source([(-3.0, ("z", "x")), (-2.0, ("y", "x"))], cache, lines)
        == f"(-{value})"
    )
    assert len(lines) == count
    assert (
        contraction_source([(1.0, ("x", "y")), (-1.0, ("y", "x"))], cache, lines)
        == "T(0)"
    )
    assert contraction_source([(1.0, ("x",))], cache, lines) == "x"
    assert contraction_source([(2.0, ())], cache, lines) == "T(2)"
    assert contraction_source([(1e-20, ("x",))], cache, lines) != "T(0)"
