"""Sparse product constants across independent functional transforms."""

import torch

from equivariantx.ops.kernels.torch.product import product, product_indices


def test_functional_constants_do_not_escape(double_precision):
    product_indices.cache_clear()
    metadata = (2, (2, 2, 2), (((0, 0, 0), 0.3), ((1, 1, 1), -0.2)))
    values = tuple(torch.randn(3, 4) for _ in range(2))
    tangents = tuple(torch.randn_like(value) for value in values)

    def function(x, y):
        return product(metadata, 2, 0, (x, y, x), x.shape[0])

    def reference(x, y):
        return x * y * x.new_tensor((0.3, 0.3, -0.2, -0.2))

    for scale in (1.0, 0.7, 1.0):
        inputs = tuple(value * scale for value in values)
        torch.testing.assert_close(
            torch.func.jvp(function, inputs, tangents),
            torch.func.jvp(reference, inputs, tangents),
        )
        torch.testing.assert_close(function(*inputs), reference(*inputs))
    product_indices.cache_clear()
