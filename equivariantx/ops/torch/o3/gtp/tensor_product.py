"""Independent quadrature and coefficient evaluation for Gaunt products."""

import torch


def transform(module, features, basis):
    return features if module.method == "coefficients" else features @ basis


def product(module, x, y, name):
    if module.method == "coefficients":
        return torch.einsum(
            "...ci,...cj,ijk->...ck", x, y, getattr(module, f"{name}_coupling")
        )
    return (x * y) @ getattr(module, f"{name}_basis")
