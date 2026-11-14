"""Ordinary alignment charts and operand-only Torch geometry adjoints."""

import torch

from ... import config
from ...common.torch import scalar


def normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps * eps)


def smooth_step(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    eps = torch.finfo(x.dtype).eps
    left = torch.exp(-1.0 / torch.clamp(x, min=eps))
    right = torch.exp(-1.0 / torch.clamp(1.0 - x, min=eps))
    interior = left / (left + right)
    return torch.where(
        x <= 0.0,
        torch.zeros_like(x),
        torch.where(x >= 1.0, torch.ones_like(x), interior),
    )


def interpolate_quaternions(
    q0: torch.Tensor,
    q1: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    blended = (1.0 - weight.unsqueeze(-1)) * q0 + weight.unsqueeze(-1) * q1
    return normalize(blended, eps)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack(
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                dim=-1,
            ),
            torch.stack(
                [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
                dim=-1,
            ),
            torch.stack(
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def alignment_quaternion(vector: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return the smooth, regularized quaternion aligning vectors with y."""
    unit_vector = normalize(vector, eps)
    x, y, z = unit_vector.unbind(dim=-1)
    q_pos = normalize(torch.stack([1.0 + y, -z, torch.zeros_like(x), x], dim=-1), eps)
    q_neg = normalize(torch.stack([-z, 1.0 - y, x, torch.zeros_like(x)], dim=-1), eps)
    blend = smooth_step(0.5 * (y + 1.0))
    return interpolate_quaternions(q_neg, q_pos, blend, eps)


def rotation_matrix_to_y_axis(
    vector: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return rotations that align nonzero vectors with the positive y-axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero three-dimensional vectors with shape ``(..., 3)``.
    eps : float, optional
        Positive regularization used by vector and quaternion normalization.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``. Applying each matrix to
        its corresponding input vector aligns the result with ``[0, 1, 0]``.

    Examples
    --------
    >>> from equivariantx import o2
    >>> vector = torch.tensor([1.0, 2.0, 3.0])
    >>> rotation = o2.rotation_matrix_to_y_axis(vector)
    >>> aligned = rotation @ vector
    >>> expected = vector.norm() * vector.new_tensor([0.0, 1.0, 0.0])
    >>> torch.testing.assert_close(aligned, expected, atol=1e-6, rtol=1e-6)
    """
    return quaternion_to_matrix(alignment_quaternion(vector, eps))


def rotation_matrix_to_x_axis(
    vector: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return rotations that align nonzero vectors with the positive x-axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero three-dimensional vectors with shape ``(..., 3)``.
    eps : float, optional
        Positive regularization used by vector and quaternion normalization.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``. Applying each matrix to
        its corresponding input vector aligns the result with ``[1, 0, 0]``.
    """
    unit_vector = normalize(vector, eps)
    x, y, z = unit_vector.unbind(dim=-1)
    q_pos = normalize(torch.stack([1.0 + x, torch.zeros_like(x), z, -y], dim=-1), eps)
    q_neg = normalize(torch.stack([-y, z, torch.zeros_like(x), 1.0 - x], dim=-1), eps)
    blend = smooth_step(0.5 * (x + 1.0))
    quaternion = interpolate_quaternions(q_neg, q_pos, blend, eps)
    return quaternion_to_matrix(quaternion)


def rotation_matrix_to_z_axis(
    vector: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return rotations that align nonzero vectors with the positive z-axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero three-dimensional vectors with shape ``(..., 3)``.
    eps : float, optional
        Positive regularization used by vector and quaternion normalization.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``. Applying each matrix to
        its corresponding input vector aligns the result with ``[0, 0, 1]``.
    """
    unit_vector = normalize(vector, eps)
    x, y, z = unit_vector.unbind(dim=-1)
    q_pos = normalize(torch.stack([1.0 + z, y, -x, torch.zeros_like(x)], dim=-1), eps)
    q_neg = normalize(torch.stack([-x, torch.zeros_like(x), 1.0 - z, y], dim=-1), eps)
    blend = smooth_step(0.5 * (z + 1.0))
    quaternion = interpolate_quaternions(q_neg, q_pos, blend, eps)
    return quaternion_to_matrix(quaternion)


def reference(vector, axis, eps=1e-7, quaternion=False):
    """Evaluate the original vectorized chart with ordinary Torch AD."""
    if quaternion:
        return alignment_quaternion(vector, eps)
    return (
        rotation_matrix_to_x_axis,
        rotation_matrix_to_y_axis,
        rotation_matrix_to_z_axis,
    )[axis](vector, eps)


def evaluate(vector, axis, eps=1e-7, quaternion=False):
    """Use the independent reference or the operand-only scalar adjoint."""
    if config.TORCH_REFERENCE or not config.TORCH_ALIGNMENT_ADJOINT:
        return reference(vector, axis, eps, quaternion)
    value = scalar.evaluate(axis, eps, quaternion, (), vector.reshape(-1, 3))[0]
    return value.reshape(*vector.shape[:-1], *((4,) if quaternion else (3, 3)))
