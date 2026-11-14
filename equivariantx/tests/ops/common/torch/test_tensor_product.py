"""Instruction grouping retains separate noncontiguous learned paths."""

from types import SimpleNamespace

import pytest
import torch

from equivariantx.ops.common.torch.tensor_product import (
    TensorProduct,
    instruction_groups,
)


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("grouped", [False, True])
def test_independent_weights(layout, grouped, double_precision):
    instructions = [
        SimpleNamespace(
            i_in1=i % 2,
            i_in2=i % 2,
            i_out=0,
            connection_mode="uuu",
            has_weight=True,
            path_shape=(2,),
            path_weight=(i + 1) / 10,
        )
        for i in range(8)
    ]
    order, groups = instruction_groups(instructions, [(0,)] * 8, grouped=grouped)
    module = TensorProduct(
        ([(2, 2)] * 2, [(2, 2)] * 2, [(2, 2)]), groups, lambda x, y, _: x * y, layout
    )
    values = tuple(
        torch.randn(*shape, requires_grad=True) for shape in ((3, 8), (3, 8), (16,))
    )
    x, y, weights = values
    expected = 0
    for i, instruction in enumerate(instructions):
        offset = 4 * (i % 2)
        product = x[:, offset : offset + 4] * y[:, offset : offset + 4]
        w = weights[2 * i : 2 * i + 2]
        w = w.repeat_interleave(2) if layout == "mul_ir" else w.repeat(2)
        expected = expected + product * w * instruction.path_weight
    actual = module(*values, torch.tensor([instructions[i].path_weight for i in order]))
    torch.testing.assert_close(actual, expected)
    ga, gb = [
        torch.autograd.grad(value.square().sum(), values)
        for value in (actual, expected)
    ]
    torch.testing.assert_close(ga, gb)
