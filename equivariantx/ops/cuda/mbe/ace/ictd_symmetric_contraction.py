"""Cartesian projection and basis conversion for CUDA contractions."""

from ....common.torch.basis import project_features
from ...co3.basis import change_basis
from ...co3.basis import project as project_blocks


def project(module, features):
    """Project flattened Cartesian features with CUDA operations."""
    return project_features(module, features, project_blocks)
