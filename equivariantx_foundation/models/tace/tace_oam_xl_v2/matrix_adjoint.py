"""Paired matrix products for mixed linear adjoints."""

import torch


def matrix_adjoint(a, b, c, d):
    """Return ``a @ b + c @ d`` for equal-shaped FP32 matrix pairs."""
    if any(x.dtype != torch.float32 or x.ndim != 2 for x in (a, b, c, d)):
        raise ValueError("Paired matrix adjoints require FP32 matrices.")
    if a.shape != c.shape or b.shape != d.shape or a.shape[1] != b.shape[0]:
        raise ValueError("Paired matrix dimensions must agree.")
    return torch.addmm(a @ b, c, d)
