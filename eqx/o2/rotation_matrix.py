################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch


def _norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps * eps)


def _quaternion_normalize(q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return q / _norm(q, eps)


def _smooth_step_cinf(x: torch.Tensor) -> torch.Tensor:
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


def _quaternion_nlerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    blended = (1.0 - weight.unsqueeze(-1)) * q0 + weight.unsqueeze(-1) * q1
    return _quaternion_normalize(blended, eps)


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
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
    """
    unit_vector = vector / _norm(vector, eps)
    x, y, z = unit_vector.unbind(dim=-1)
    q_pos = _quaternion_normalize(
        torch.stack([1.0 + y, -z, torch.zeros_like(x), x], dim=-1), eps
    )
    q_neg = _quaternion_normalize(
        torch.stack([-z, 1.0 - y, x, torch.zeros_like(x)], dim=-1), eps
    )
    blend = _smooth_step_cinf(0.5 * (y + 1.0))
    quaternion = _quaternion_nlerp(q_neg, q_pos, blend, eps)
    return _quaternion_to_rotation_matrix(quaternion)


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
    unit_vector = vector / _norm(vector, eps)
    x, y, z = unit_vector.unbind(dim=-1)
    q_pos = _quaternion_normalize(
        torch.stack([1.0 + x, torch.zeros_like(x), z, -y], dim=-1), eps
    )
    q_neg = _quaternion_normalize(
        torch.stack([-y, z, torch.zeros_like(x), 1.0 - x], dim=-1), eps
    )
    blend = _smooth_step_cinf(0.5 * (x + 1.0))
    quaternion = _quaternion_nlerp(q_neg, q_pos, blend, eps)
    return _quaternion_to_rotation_matrix(quaternion)


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
    unit_vector = vector / _norm(vector, eps)
    x, y, z = unit_vector.unbind(dim=-1)
    q_pos = _quaternion_normalize(
        torch.stack([1.0 + z, y, -x, torch.zeros_like(x)], dim=-1), eps
    )
    q_neg = _quaternion_normalize(
        torch.stack([-x, torch.zeros_like(x), 1.0 - z, y], dim=-1), eps
    )
    blend = _smooth_step_cinf(0.5 * (z + 1.0))
    quaternion = _quaternion_nlerp(q_neg, q_pos, blend, eps)
    return _quaternion_to_rotation_matrix(quaternion)
