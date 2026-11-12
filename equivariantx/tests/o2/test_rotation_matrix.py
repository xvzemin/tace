"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize(
    ("rotation_matrix", "axis"),
    [
        (o2.rotation_matrix_to_x_axis, 0),
        (o2.rotation_matrix_to_y_axis, 1),
        (o2.rotation_matrix_to_z_axis, 2),
    ],
)
def test_rotation_matrix_to_axis(rotation_matrix, axis):
    generator = torch.Generator().manual_seed(20260903)
    vectors = torch.randn(64, 3, dtype=DTYPE, generator=generator).to(DEVICE)
    rotation = rotation_matrix(vectors)
    rotated = torch.einsum("bij,bj->bi", rotation, vectors)
    expected = torch.zeros_like(vectors)
    expected[:, axis] = torch.linalg.vector_norm(vectors, dim=-1)

    torch.testing.assert_close(rotated, expected)
    identity = torch.eye(3, dtype=DTYPE, device=DEVICE).expand_as(rotation)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity)
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones(vectors.size(0), dtype=DTYPE, device=DEVICE),
    )
