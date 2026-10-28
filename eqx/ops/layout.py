"""Indexed feature sums and their transposes."""

import torch


@torch.library.custom_op("eqx::gather_sum", mutates_args=(), device_types="cuda")
def gather_sum(
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
    from ..backends import get_kernel

    return get_kernel("gather_sum", "cuda")(features, indices)


@gather_sum.register_fake
def gather_sum_fake(features, indices):
    rows = features[0].size(0) if indices[0] is None else indices[0].numel()
    return features[0].new_empty((rows, features[0].size(1)))


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


gather_sum.register_autograd(
    gather_sum_backward, setup_context=gather_sum_setup_context
)


@torch.library.custom_op("eqx::grouped_permute", mutates_args=(), device_types="cuda")
def grouped_permute(
    features: torch.Tensor,
    indices: torch.Tensor,
    inverse: torch.Tensor,
    widths: str,
    transpose: bool = False,
) -> torch.Tensor:
    """Pack feature groups into contiguous batch matrices, or apply its transpose."""
    from ..backends import get_kernel

    return get_kernel("grouped_permute", "cuda")(
        features, indices, inverse, widths, transpose
    )


@grouped_permute.register_fake
def grouped_permute_fake(features, indices, inverse, widths, transpose=False):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


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


grouped_permute.register_autograd(
    permutation_backward, setup_context=permutation_setup_context
)


@torch.library.custom_op("eqx::indexed_sum", mutates_args=(), device_types="cuda")
def indexed_sum(
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
    from ..backends import get_kernel

    return get_kernel("indexed_sum", "cuda")(
        features, indices, ptr, transpose_indices, transpose_ptr, count, transpose
    )


@indexed_sum.register_fake
def indexed_sum_fake(
    features, indices, ptr, transpose_indices, transpose_ptr, count, transpose=False
):
    return features.new_empty((features.size(0), ptr.numel() - 1))


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


indexed_sum.register_autograd(backward, setup_context=setup_context)
