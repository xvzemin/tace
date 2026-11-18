"""Cartesian delta/epsilon products and their exact sparse coefficients."""

import math
from functools import lru_cache
from itertools import product

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config


def coupling_size(l1, l2, l3):
    """Number of nonzero delta/epsilon terms before STF projection."""
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError("Degrees must satisfy the triangle inequality.")
    k, odd = divmod(l1 + l2 - l3, 2)
    return (1 + odd) * 3 ** (l3 + k)


def literal_coupling(degrees, budget):
    """Bound both source size and the longest reduction over any leg."""
    return coupling_size(*degrees) <= min(budget, 64 * 3 ** min(degrees))


def inline_reduction(x, degrees):
    l1, l2, l3 = degrees
    return (
        config.CARTESIAN_REDUCTION_INLINE
        and x.is_cuda
        and ((l1 + l2 - l3) // 2 <= 2 or l3 == 0)
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


def reduced_product(x, y, l1, l2, l3):
    """Expose a delta or epsilon product and its contracted axis to fusion."""
    k, odd = divmod(l1 + l2 - l3, 2)
    a, d, b = 3 ** (l1 - k - odd), 3**k, 3 ** (l2 - k - odd)
    p = 3 if odd else 1
    x = x.unflatten(-1, (a, p, d)).unsqueeze(-1)
    y = y.unflatten(-1, (d, p, b)).transpose(-3, -2).unsqueeze(-4)
    if odd:
        x, y = torch.broadcast_tensors(x, y)
        value = torch.linalg.cross(x, y, dim=-3)
    else:
        value = x * y
    return value.sum(-2).flatten(-3) * ((1 + odd) * d) ** -0.5


def evaluate(x, y, l1, l2, l3):
    """Contract with a bounded scalar reduction or a compiler-visible product."""
    degrees = (l1, l2, l3)
    if inline_reduction(x, degrees) and not literal_coupling(
        degrees, config.CARTESIAN_LITERAL_TERMS
    ):
        return reduced_product(x, y, *degrees)
    if (
        x.is_cuda
        and l3 == 0
        and l1 == l2 >= 4
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and x.numel() * x.element_size() <= config.TORCH_CARTESIAN_DOT_BYTES
    ):
        return (x * y).sum(-1, keepdim=True) * (3**l1) ** -0.5
    return cartesian_product(x, y, *degrees)


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
