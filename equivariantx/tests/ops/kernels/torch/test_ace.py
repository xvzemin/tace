"""Ordinary sparse ACE schedules and every transposed factor role."""

import pytest
import torch

from equivariantx.ops import get_kernel
from equivariantx.ops.polynomial import polynomial_adjoint
from equivariantx.utils.metadata import encode_plans


@pytest.mark.parametrize("kind", ["unweighted", "weighted", "gated"])
@pytest.mark.parametrize("nodes", [0, 5])
@pytest.mark.parametrize("shared", [False, True])
def test_products_and_transposes(kind, nodes, shared, double_precision):
    x, y = [torch.randn(nodes, 3, requires_grad=True) for _ in range(2)]
    types = torch.arange(nodes) % 2
    weighted = kind != "unweighted"
    output_dim = 2 if weighted else 3
    operands = [x, y]
    record = (0, 0, 0, 0, 3, 2 if weighted else 0, 1, 1, 1, 0)
    dims = [3, 3]
    expected = x * y
    continuous = [x, y]
    if weighted:
        weight = torch.randn(1 if shared else 2, 6, requires_grad=True)
        operands.append(weight)
        continuous.append(weight)
        dims.append(6)
    operands.append(x.new_empty((0, output_dim)))
    dims.append(output_dim)
    role = len(operands) - 1
    if kind == "gated":
        gates = torch.randn(nodes, 3, requires_grad=True)
        operands.append(gates)
        continuous.append(gates)
        dims.append(3)
        expected = expected * gates
    if weighted:
        indices = torch.zeros_like(types) if shared else types
        matrix = weight[indices].reshape(nodes, 3, 2)
        expected = (expected.unsqueeze(1) @ matrix).squeeze(1)
    expected = expected * 0.7
    schedules = tuple((((0, 1, 1), ((record, 0.7),)),) for _ in operands)
    metadata = repr((tuple(dims), weighted, schedules))
    mapping = tuple(range(len(operands)))
    program = repr(((mapping, role, 0),))
    backend_kind = "ace"
    name = "ace_contract"
    if kind == "gated":
        metadata = encode_plans([metadata])
        program = repr(((0, mapping, role, 0),))
        backend_kind, name = "bilinear", "bilinear_ace"
    evaluate = get_kernel(name, "torch")
    actual = evaluate(metadata, program, types, operands)[0]
    torch.testing.assert_close(actual, expected)
    seed = torch.randn_like(expected)
    adjoint, values, destinations = polynomial_adjoint(
        program, operands, (seed,), [x.requires_grad for x in operands], backend_kind
    )
    result = evaluate(metadata, adjoint, types, values)
    gradients = torch.autograd.grad(expected, continuous, seed, create_graph=True)
    for value, gradient in zip(continuous, gradients):
        index = next(i for i, operand in enumerate(operands) if operand is value)
        torch.testing.assert_close(result[destinations[index]], gradient)
    for _ in range(4):
        actual, expected = [
            torch.cat(
                [
                    gradient.flatten()
                    for gradient in torch.autograd.grad(
                        value.sin().sum(), continuous, create_graph=True
                    )
                ]
            )
            / 10
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
