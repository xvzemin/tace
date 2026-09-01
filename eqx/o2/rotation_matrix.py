################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
The rotation matrix constructed here aligns the edge with the y-axis [0,1,0].

For details on obtaining a rotation matrix from quaternions, see:
https://en.wikipedia.org/wiki/Hopf_fibration#Explicit_formulae
and
https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/common/quaternion/quaternion_utils.py
"""

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


def init_edge_rot_mat_quaternion(
    edge_distance_vec: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Construct smooth rotation matrices aligned with edge vectors.

    Parameters
    ----------
    edge_distance_vec : torch.Tensor
        Three-dimensional vectors with shape ``(..., 3)``. Each resulting
        matrix aligns its vector with the positive second Cartesian axis.
    eps : float, optional
        Positive regularization used by vector and quaternion normalization.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.

    Notes
    -----
    Two quaternion charts are blended smoothly to avoid a singular choice of
    frame near either direction of the alignment axis.
    """
    edge_unit = edge_distance_vec / _norm(edge_distance_vec, eps)
    x, y, z = edge_unit.unbind(dim=-1)
    q_pos = _quaternion_normalize(
        torch.stack([1.0 + y, -z, torch.zeros_like(x), x], dim=-1), eps
    )
    q_neg = _quaternion_normalize(
        torch.stack([-z, 1.0 - y, x, torch.zeros_like(x)], dim=-1), eps
    )
    blend = _smooth_step_cinf(0.5 * (y + 1.0))
    quaternion = _quaternion_nlerp(q_neg, q_pos, blend, eps)
    return _quaternion_to_rotation_matrix(quaternion)
