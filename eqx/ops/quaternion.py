"""CUDA quaternion Wigner matrices and their directional derivatives."""

import torch


@torch.library.custom_op(
    "eqx::quaternion_polynomial", mutates_args=(), device_types="cuda"
)
def quaternion_polynomial(
    lmax: int, transpose: bool, values: list[torch.Tensor]
) -> torch.Tensor:
    """Contract Wigner polynomial derivatives without materializing Jacobians."""
    from ..backends import get_kernel

    return get_kernel("quaternion_polynomial", "cuda")(lmax, transpose, values)


@quaternion_polynomial.register_fake
def quaternion_polynomial_fake(lmax, transpose, values):
    width = 4 if transpose else sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    return values[0].new_empty((values[0].size(0), width))


def polynomial_setup_context(ctx, inputs, output):
    ctx.lmax, ctx.transpose, values = inputs
    ctx.save_for_backward(*values)


def polynomial_backward(ctx, gradient):
    q, *values = ctx.saved_tensors
    directions = values[:-1] if ctx.transpose else values
    cotangent = values[-1] if ctx.transpose else gradient
    extra = [gradient] if ctx.transpose else []
    results = []
    for i, need in enumerate(ctx.needs_input_grad[2]):
        if not need:
            results.append(None)
        elif ctx.transpose and i == len(values):
            results.append(
                quaternion_polynomial(ctx.lmax, False, [q, *directions, gradient])
            )
        else:
            selected = [v for j, v in enumerate(directions, 1) if i == 0 or j != i]
            results.append(
                quaternion_polynomial(ctx.lmax, True, [q, *selected, *extra, cotangent])
            )
    return None, None, results


quaternion_polynomial.register_autograd(
    polynomial_backward, setup_context=polynomial_setup_context
)


def quaternion_wigner(vectors, lmax):
    """Return packed Wigner matrices using direct quaternion polynomials.

    Parameters
    ----------
    vectors : torch.Tensor
        Nonzero frame directions of shape ``(edges, 3)`` on CUDA, in float32
        or float64. Their alignment is the same as the recursive method.
    lmax : int
        Maximum angular degree. All orders are retained.

    Returns
    -------
    torch.Tensor
        Degree matrices concatenated as ``(edges, sum((2*l+1)**2))``.
        Both the alignment and polynomial contractions support higher derivatives.
    """
    from .wigner import alignment_cuda

    if not vectors.is_cuda or vectors.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "Quaternion Wigner matrices require CUDA float32 or float64 inputs."
        )
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (edges, 3).")
    if lmax == 0:
        return vectors.new_ones((vectors.size(0), 1))
    if lmax == 1:
        return alignment_cuda(repr("wigner1"), [vectors])[0]
    q = alignment_cuda(repr("quaternion"), [vectors])[0]
    return quaternion_polynomial(lmax, False, [q])
