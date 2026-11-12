"""CUDA quaternion Wigner matrices and their directional derivatives."""

import torch

from ..utils.outputs import quaternion_polynomial_output
from .precision import validate_uniform


@torch.library.custom_op(
    "eqx::quaternion_polynomial", mutates_args=(), device_types="cuda"
)
def quaternion_polynomial_cuda(
    lmax: int, transpose: bool, values: list[torch.Tensor]
) -> torch.Tensor:
    """Contract Wigner polynomial derivatives without materializing Jacobians."""
    from . import get_kernel

    return get_kernel("quaternion_polynomial", "cuda")(lmax, transpose, values)


quaternion_polynomial_cuda.register_fake(quaternion_polynomial_output)


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


quaternion_polynomial_cuda.register_autograd(
    polynomial_backward, setup_context=polynomial_setup_context
)


@torch.compiler.allow_in_graph
def quaternion_polynomial(lmax, transpose, values):
    """Expose directional polynomial expressions to functional transformations."""
    validate_uniform(values)
    if torch._C._are_functorch_transforms_active():
        from .kernels.torch.quaternion import quaternion_polynomial as evaluate

        return evaluate(lmax, transpose, values)
    return torch.ops.eqx.quaternion_polynomial(lmax, transpose, values)


def quaternion_wigner(vectors, lmax, *, backend="torch"):
    """Return packed Wigner matrices using direct quaternion polynomials.

    Parameters
    ----------
    vectors : torch.Tensor
        Nonzero frame directions of shape ``(edges, 3)`` in float32 or float64.
    lmax : int
        Maximum angular degree. All orders are retained.
    backend : {"torch", "cuda"}, optional
        Execution provider. CPU inputs use Torch on either setting.

    Returns
    -------
    torch.Tensor
        Degree matrices concatenated as ``(edges, sum((2*l+1)**2))``.
        Both the alignment and polynomial contractions support higher derivatives.
    """
    from .precision import validate_precision
    from .wigner import alignment_cuda

    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    validate_precision((vectors,))
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (edges, 3).")
    if backend == "torch" or not vectors.is_cuda:
        from .kernels.torch.quaternion import quaternion_wigner as evaluate_torch

        return evaluate_torch(vectors, lmax)
    if lmax == 0:
        return vectors.new_ones((vectors.size(0), 1))
    if lmax == 1:
        return alignment_cuda(repr("wigner1"), [vectors])[0]
    q = alignment_cuda(repr("quaternion"), [vectors])[0]
    return quaternion_polynomial(lmax, False, [q])
