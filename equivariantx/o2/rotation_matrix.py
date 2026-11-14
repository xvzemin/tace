"""Smooth O(2) frame alignment charts."""

import torch

from ..ops import resolve_backend
from ..ops.precision import validate_precision


def execute(vector, axis, eps, quaternion, backend):
    validate_precision((vector,))
    if vector.shape[-1] != 3:
        raise ValueError("Alignment vectors must have three components.")
    if backend == "cuda":
        if not vector.is_cuda:
            raise ValueError("backend='cuda' requires CUDA vectors.")
        from ..ops.cuda.o2.rotation_matrix import evaluate
    else:
        from ..ops.torch.o2.rotation_matrix import evaluate
    return evaluate(vector, axis, eps, quaternion)


def alignment_quaternion(
    vector: torch.Tensor, eps: float = 1e-7, *, backend=None
) -> torch.Tensor:
    """Return the regularized quaternion aligning vectors with y.

    Parameters
    ----------
    vector : torch.Tensor
        Vectors with shape ``(..., 3)``.
    eps : float, optional
        Normalization regularization.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider.

    Returns
    -------
    torch.Tensor
        Quaternions of shape ``(..., 4)`` in ``(w, x, y, z)`` order.
    """
    backend = resolve_backend(backend, vector.device)
    return execute(vector, 1, eps, True, backend)


def rotation_matrix_to_y_axis(
    vector: torch.Tensor, eps: float = 1e-7, *, backend=None
) -> torch.Tensor:
    """Return smooth rotations aligning vectors with the positive y-axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero vectors with shape ``(..., 3)``.
    eps : float, optional
        Normalization regularization.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.

    Examples
    --------
    >>> from equivariantx import o2
    >>> vector = torch.tensor([1.0, 2.0, 3.0])
    >>> rotation = o2.rotation_matrix_to_y_axis(vector)
    >>> torch.testing.assert_close(rotation @ vector, vector.norm() * vector.new_tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=1e-6)
    """
    backend = resolve_backend(backend, vector.device)
    return execute(vector, 1, eps, False, backend)


def rotation_matrix_to_x_axis(
    vector: torch.Tensor, eps: float = 1e-7, *, backend=None
) -> torch.Tensor:
    """Return smooth rotations aligning vectors with the positive x-axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero vectors with shape ``(..., 3)``.
    eps : float, optional
        Normalization regularization.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.
    """
    backend = resolve_backend(backend, vector.device)
    return execute(vector, 0, eps, False, backend)


def rotation_matrix_to_z_axis(
    vector: torch.Tensor, eps: float = 1e-7, *, backend=None
) -> torch.Tensor:
    """Return smooth rotations aligning vectors with the positive z-axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero vectors with shape ``(..., 3)``.
    eps : float, optional
        Normalization regularization.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.
    """
    backend = resolve_backend(backend, vector.device)
    return execute(vector, 2, eps, False, backend)
