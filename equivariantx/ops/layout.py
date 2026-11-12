"""Indexed feature sums and their transposes."""

import torch
from .precision import validate_uniform

from ..utils.outputs import (
    gather_sum_output,
    grouped_permute_output,
    indexed_sum_output,
)


@torch.library.custom_op("eqx::gather_sum", mutates_args=(), device_types="cuda")
def gather_sum_cuda(
    features: list[torch.Tensor],
    indices: list[torch.Tensor | None],
) -> torch.Tensor:
    """Sum feature matrices after optional row gathers.

    Parameters
    ----------
    features : list of torch.Tensor
        Matrices with the same channel width, dtype and device.
    indices : list of torch.Tensor or None
        Row indices for each matrix. ``None`` leaves its rows unchanged.
    """
    from . import get_kernel

    return get_kernel("gather_sum", "cuda")(features, indices)


gather_sum_cuda.register_fake(gather_sum_output)


def gather_sum_setup_context(ctx, inputs, output):
    features, indices = inputs
    ctx.shapes = [value.shape for value in features]
    ctx.save_for_backward(*indices)


def gather_sum_backward(ctx, gradient):
    indices = ctx.saved_tensors
    gradients = [
        None
        if not required
        else gradient
        if index is None
        else gradient.new_zeros(shape).index_add(0, index, gradient)
        for shape, index, required in zip(ctx.shapes, indices, ctx.needs_input_grad[0])
    ]
    return gradients, [None] * len(indices) if all(
        i is not None for i in indices
    ) else None


gather_sum_cuda.register_autograd(
    gather_sum_backward, setup_context=gather_sum_setup_context
)


@torch.library.custom_op("eqx::grouped_permute", mutates_args=(), device_types="cuda")
def grouped_permute_cuda(
    features: torch.Tensor,
    indices: torch.Tensor,
    inverse: torch.Tensor,
    widths: str,
    transpose: bool = False,
) -> torch.Tensor:
    """Pack feature groups into contiguous batch matrices, or apply its transpose."""
    from . import get_kernel

    return get_kernel("grouped_permute", "cuda")(
        features, indices, inverse, widths, transpose
    )


grouped_permute_cuda.register_fake(grouped_permute_output)


def permutation_setup_context(ctx, inputs, output):
    _, indices, inverse, ctx.widths, ctx.transpose = inputs
    ctx.save_for_backward(inverse, indices)


def permutation_backward(ctx, gradient):
    inverse, indices = ctx.saved_tensors
    return (
        grouped_permute(gradient, inverse, indices, ctx.widths, not ctx.transpose),
        None,
        None,
        None,
        None,
    )


grouped_permute_cuda.register_autograd(
    permutation_backward, setup_context=permutation_setup_context
)


@torch.library.custom_op("eqx::indexed_sum", mutates_args=(), device_types="cuda")
def indexed_sum_cuda(
    features: torch.Tensor,
    indices: torch.Tensor,
    ptr: torch.Tensor,
    transpose_indices: torch.Tensor,
    transpose_ptr: torch.Tensor,
    count: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
    """Sum indexed columns without an expanded gather or atomic additions.

    Parameters
    ----------
    features : torch.Tensor
        Input matrix of shape ``(batch, features)``.
    indices, ptr : torch.Tensor
        Column indices and row pointers of the binary CSR map.
    transpose_indices, transpose_ptr : torch.Tensor
        CSR indices of its transpose, used by recursive backward passes.
    count : torch.Tensor
        Row divisors, or an empty tensor for unnormalized sums.
    transpose : bool, optional
        Apply divisors to inputs instead of outputs in a transposed map.
    """
    from . import get_kernel

    return get_kernel("indexed_sum", "cuda")(
        features, indices, ptr, transpose_indices, transpose_ptr, count, transpose
    )


indexed_sum_cuda.register_fake(indexed_sum_output)


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:-1])
    ctx.transpose = inputs[-1]


def backward(ctx, gradient):
    indices, ptr, transpose_indices, transpose_ptr, count = ctx.saved_tensors
    return (
        indexed_sum(
            gradient,
            transpose_indices,
            transpose_ptr,
            indices,
            ptr,
            count,
            not ctx.transpose,
        ),
        None,
        None,
        None,
        None,
        None,
        None,
    )


indexed_sum_cuda.register_autograd(backward, setup_context=setup_context)


@torch.compiler.allow_in_graph
def gather_sum(features, indices):
    """Sum gathered features with native execution and functional derivatives."""
    validate_uniform(features)
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.layout import gather_sum as evaluate

        return evaluate(features, indices)
    return torch.ops.eqx.gather_sum(features, indices)


@torch.compiler.allow_in_graph
def grouped_permute(features, indices, inverse, widths, transpose=False):
    """Pack feature groups with ordinary functional transformation rules."""
    validate_uniform((features,))
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.layout import grouped_permute as evaluate

        return evaluate(features, indices, inverse, widths, transpose)
    return torch.ops.eqx.grouped_permute(features, indices, inverse, widths, transpose)


@torch.compiler.allow_in_graph
def indexed_sum(
    features, indices, ptr, transpose_indices, transpose_ptr, count, transpose=False
):
    """Sum CSR feature columns with native execution and functional derivatives."""
    validate_uniform((features,))
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.layout import indexed_sum as evaluate

        return evaluate(
            features, indices, ptr, transpose_indices, transpose_ptr, count, transpose
        )
    return torch.ops.eqx.indexed_sum(
        features, indices, ptr, transpose_indices, transpose_ptr, count, transpose
    )
