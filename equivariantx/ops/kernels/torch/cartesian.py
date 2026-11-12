"""Cartesian delta/epsilon products and their exact sparse coefficients."""

import math
from functools import lru_cache
from itertools import product

import torch


def cartesian_product(x, y, l1, l2, l3):
    """Return the unprojected Cartesian product with trailing size 3**l3."""
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError("Degrees must satisfy the triangle inequality.")
    k, odd = divmod(l1 + l2 - l3, 2)
    if not k:
        if not odd:
            if not l1 or not l2:
                return x * y
            return (x.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2)
        x = x.reshape(*x.shape[:-1], 3 ** (l1 - 1), 1, 3)
        y = y.reshape(*y.shape[:-1], 3, 3 ** (l2 - 1))
        y = y.transpose(-2, -1).unsqueeze(-3)
        x, y = torch.broadcast_tensors(x, y)
        value = torch.linalg.cross(x, y).transpose(-2, -1)
        return value.flatten(-3) / math.sqrt(2)
    if not odd:
        x = x.reshape(*x.shape[:-1], 3 ** (l1 - k), 3**k)
        y = y.reshape(*y.shape[:-1], 3**k, 3 ** (l2 - k))
        return (x @ y).flatten(-2) / math.sqrt(3**k)
    x = x.reshape(*x.shape[:-1], 3 ** (l1 - k - 1), 3, 3**k)
    y = y.reshape(*y.shape[:-1], 3**k, 3, 3 ** (l2 - k - 1))
    values = [
        x[..., a, :] @ y[..., b, :] - x[..., b, :] @ y[..., a, :]
        for a, b in ((1, 2), (2, 0), (0, 1))
    ]
    return torch.stack(values, dim=-2).flatten(-3) / math.sqrt(2 * 3**k)


@lru_cache(maxsize=None)
def coupling_coefficients(l1, l2, l3):
    """Return integer delta/epsilon coefficients before normalization."""
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError("Degrees must satisfy the triangle inequality.")
    k, odd = divmod(l1 + l2 - l3, 2)
    left, contracted, right = 3 ** (l1 - k - odd), 3**k, 3 ** (l2 - k - odd)
    entries = []
    for i, j, t in product(range(left), range(right), range(contracted)):
        if odd:
            for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
                out = (i * 3 + c) * right + j
                entries.append(
                    ((i * 3 + a) * contracted + t, (t * 3 + b) * right + j, out, 1)
                )
                entries.append(
                    ((i * 3 + b) * contracted + t, (t * 3 + a) * right + j, out, -1)
                )
        else:
            entries.append((i * contracted + t, t * right + j, i * right + j, 1))
    return tuple(entries)


@lru_cache(maxsize=None)
def normalized_coefficients(l1, l2, l3):
    """Return coefficients with the same normalization as cartesian_product."""
    traces, odd = divmod(l1 + l2 - l3, 2)
    scale = math.sqrt(3**traces * (2 if odd else 1))
    return tuple(
        (a, b, c, value / scale) for a, b, c, value in coupling_coefficients(l1, l2, l3)
    )
