"""Analytic direction programs against ordinary automatic differentiation."""

import pytest
import torch

from equivariantx.ops.kernels.torch.edge import evaluate_torch
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.program import Program, next_adjoint, next_tangent


@pytest.mark.parametrize(
    "operation", ["exp", "sin", "cos", "tanh", "sigmoid", "silu", "reciprocal", "rsqrt"]
)
@pytest.mark.parametrize("reduction", ["edge", "shared", "target"])
def test_direction_program(operation, reduction):
    program = Program()
    x = program.input(0, "source", 4)
    y = program.unary(operation, x)
    gathered = program.gather(y, (1, 3, 1))
    y = program.scatter(gathered, (3, 0, 2), 4)
    metadata = repr((tuple(program.nodes), ((y, 0, reduction),)))
    source, target = torch.tensor([[0, 2, 1, 0], [0, 1, 1, 0]])
    x = torch.rand(3, 4, dtype=torch.float64) + 1.0
    direction = torch.randn_like(x)

    def function(x):
        return evaluate_torch(metadata, (x,), source, target, 2)[0]

    tangent_program = next_tangent(metadata, 1)
    nodes, _ = parse_metadata(tangent_program)
    assert all(data[0] < 2 for op, _, _, data in nodes if op == "input")
    result = evaluate_torch(tangent_program, (x, direction), source, target, 2)[0]
    expected = torch.func.jvp(function, (x,), (direction,))[1]
    torch.testing.assert_close(result, expected, atol=1e-11, rtol=1e-11)


def test_constant_and_repeated_output():
    program = Program()
    x = program.input(0, "edge", 2)
    square = program.binary("mul", x, x)
    constant = program.constant(2, 4)
    metadata = repr(
        (
            tuple(program.nodes),
            ((x, 0, "edge"), (square, 0, "edge"), (constant, 1, "shared")),
        )
    )
    index = torch.arange(3)
    x, direction = torch.randn(3, 2), torch.randn(3, 2)
    result = evaluate_torch(next_tangent(metadata, 1), (x, direction), index, index, 3)
    torch.testing.assert_close(result, [(1 + 2 * x) * direction, torch.zeros(1, 2)])


@pytest.mark.parametrize("operation", ["product", "frame"])
@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("transpose", [False, True])
def test_multilinear_direction_program(operation, role, transpose):
    program = Program()
    operands = tuple(program.input(i, "edge", 4) for i in range(3))
    if operation == "product":
        description = 2, (2, 2, 2), (((0, 1, 0), 0.7), ((1, 0, 1), -0.3))
        root = program.product(operands, description, role)
    else:
        layout = repr((4, 4, 4, ((0, 2, 2, 0, ((0, 0, 1), (2, 1, -1))),)))
        root = program.add("frame", 4, operands, (layout, role))
    metadata = repr((tuple(program.nodes), ((root, 0, "edge"),)))
    values = tuple(torch.randn(3, 4, dtype=torch.float64) for _ in range(3))
    if transpose:
        metadata = next_adjoint(metadata, tuple(range(3)), (0,), len(values))
        values += (torch.randn(3, 4, dtype=torch.float64),)
    directions = tuple(torch.randn_like(value) for value in values)
    index = torch.arange(3)
    tangent = next_tangent(metadata, len(values))
    nodes, _ = parse_metadata(tangent)
    assert all(data[0] < 2 * len(values) for op, _, _, data in nodes if op == "input")

    def function(*values):
        return evaluate_torch(metadata, values, index, index, 3, optimized=True)

    actual = evaluate_torch(
        tangent, values + directions, index, index, 3, optimized=True
    )
    expected = torch.func.jvp(function, values, directions)[1]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
