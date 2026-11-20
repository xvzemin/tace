"""Uniform-grid cubic polynomials."""

import torch


def evaluate(x, coefficients, step, order):
    n = coefficients.shape[0]
    inside = (x >= 0) & (x < n * step)
    r = torch.where(inside, x, 0.0)
    index = (r / step).long().clamp(max=n - 1)
    dx = (r - index.to(x.dtype) * step).unsqueeze(-1)
    a, b, c, d = coefficients[index].unbind(1)
    if order == 0:
        y = ((a * dx + b) * dx + c) * dx + d
    elif order == 1:
        y = (3 * a * dx + 2 * b) * dx + c
    elif order == 2:
        y = 6 * a * dx + 2 * b
    else:
        y = 6 * a
    return torch.where(inside[:, None], y, 0.0)
