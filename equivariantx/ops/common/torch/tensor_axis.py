"""Broadcast matrix actions along one tensor axis with exact adjoints."""

import torch

from ....utils.autograd import nested_forward_ad, torch_gradient_mask


def reference(features, matrix, left, right):
    """Contract an axis using ordinary einsum and automatic differentiation."""
    value = features.unflatten(-1, (left, matrix.shape[-1], right))
    return torch.einsum("...oi,...aib->...aob", matrix, value).flatten(-3)


def primal(features, matrix, left, right):
    """Expose small matrix axes as products and sums for joint fusion."""
    if max(matrix.shape[-2:]) > 4:
        return reference(features, matrix, left, right)
    value = features.unflatten(-1, (left, matrix.shape[-1], right))
    rows = []
    for j in range(matrix.shape[-2]):
        terms = [
            value[..., :, i, :] * matrix[..., j, i, None, None]
            for i in range(matrix.shape[-1])
        ]
        rows.append(sum(terms[1:], terms[0]))
    return torch.stack(rows, -2).flatten(-3)


def adjoint(features, matrix, gradient, left, right, needed):
    """Reduce matrix and feature cotangents over all broadcast dimensions."""
    dx = (
        primal(gradient, matrix.transpose(-1, -2), left, right).sum_to_size(
            features.shape
        )
        if needed[0]
        else None
    )
    dm = None
    if needed[1]:
        x = features.unflatten(-1, (left, matrix.shape[-1], right))
        g = gradient.unflatten(-1, (left, matrix.shape[-2], right))
        if max(matrix.shape[-2:]) <= 4:
            rows = [
                torch.stack(
                    [
                        (g[..., :, j, :] * x[..., :, i, :]).sum((-1, -2))
                        for i in range(matrix.shape[-1])
                    ],
                    -1,
                )
                for j in range(matrix.shape[-2])
            ]
            dm = torch.stack(rows, -2)
        else:
            dm = torch.einsum("...aob,...aib->...oi", g, x)
        dm = dm.sum_to_size(matrix.shape)
    return dx, dm


@torch.compiler.allow_in_graph
class TensorAxis(torch.autograd.Function):
    """Retain original operands and differentiate the matrix action exactly."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, matrix, left, right):
        return primal(features, matrix, left, right)

    @staticmethod
    def setup_context(ctx, inputs, output):
        features, matrix, ctx.left, ctx.right = inputs
        ctx.save_for_backward(features, matrix)
        ctx.save_for_forward(features, matrix)

    @staticmethod
    def backward(ctx, gradient):
        features, matrix = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[:2])
        return (
            *adjoint(features, matrix, gradient, ctx.left, ctx.right, needed),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, dx, dm, unused_left, unused_right):
        features, matrix = ctx.saved_tensors
        dx = torch.zeros_like(features) if dx is None else dx
        dm = torch.zeros_like(matrix) if dm is None else dm
        return primal(dx, matrix, ctx.left, ctx.right) + primal(
            features, dm, ctx.left, ctx.right
        )


@torch.compiler.allow_in_graph
def evaluate(features, matrix, left=1, right=1):
    """Expose the complete handwritten adjoint to AOTAutograd."""
    if nested_forward_ad():
        return primal(features, matrix, left, right)
    return TensorAxis.apply(features, matrix, left, right)


def transform(tensor, matrix, rank, contract=reference):
    """Apply a rectangular map to each index without a Kronecker matrix."""
    outer, inner = matrix.shape[-2:]
    for axis in range(rank):
        tensor = contract(tensor, matrix, inner ** (rank - axis - 1), outer**axis)
    return tensor


def select():
    """Expose short axes to joint compilation; retain eager GEMM scheduling."""
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    from ... import config

    if config.TORCH_REFERENCE or not config.TORCH_TENSOR_AXIS:
        return reference
    if config.TORCH_TENSOR_AXIS_ADJOINT:
        return evaluate
    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        return primal
    return reference
