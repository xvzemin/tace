"""Shared, absent and repeated factors in sparse channel products."""

import pytest
import torch

from equivariantx.ops.common.torch.channel_product import channel_product, reference
from equivariantx.utils.metadata import channel_product_layout
from equivariantx.ops.common.polynomial import polynomial_adjoint


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rows", [0, 5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_polynomial_transposes(strided, rows, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    metadata = repr(
        (
            3,
            (2, 2, 1, 2),
            (
                ((0, 0, 0, 0), 1.0),
                ((1, 1, 0, 0), 2.0),
                ((0, -1, -1, 1), 0.5),
                ((1, 0, -1, 1), -0.3),
            ),
        )
    )
    if strided:
        dimensions, entries = channel_product_layout(metadata)
        metadata = repr((0, dimensions, entries))
    program = repr((((0, 1, 2, 3), 3, 0, 0),))
    values = [
        torch.randn(n, width, device=device, requires_grad=True)
        for n, width in ((rows, 6), (1, 6), (1, 3))
    ]
    operands = [*values, values[0].new_empty((0, 6))]
    x, y, w = values
    expected = torch.cat(
        (
            x[:, :3] * y[:, :3] * w + 2 * x[:, 3:] * y[:, 3:] * w,
            0.5 * x[:, :3] - 0.3 * x[:, 3:] * y[:, :3],
        ),
        1,
    )
    actual = channel_product(metadata, program, operands)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(reference(metadata, program, operands)[0], expected)
    seed = torch.randn_like(actual)
    adjoint, inputs, destinations = polynomial_adjoint(
        program, operands, [seed], (True, True, True, False)
    )
    result = channel_product(metadata, adjoint, inputs)
    gradients = torch.autograd.grad(expected, values, seed, create_graph=True)
    for index, gradient in enumerate(gradients):
        torch.testing.assert_close(result[destinations[index]], gradient)
    a, b = actual.sin().sum(), expected.sin().sum()
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value, values, create_graph=True) for value in (a, b)
        ]
        torch.testing.assert_close(ga, gb)
        a, b = [sum(value.sin().sum() for value in gradients) for gradients in (ga, gb)]
