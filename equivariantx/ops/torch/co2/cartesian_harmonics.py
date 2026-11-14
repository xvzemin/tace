"""Exact planar Cartesian harmonics without dense basis multiplication."""

import math
from functools import lru_cache

import torch

from ... import config
from ..o2 import circular_harmonics


def scale(order, normalization):
    """Return the orthonormal circular-to-Cartesian harmonic scale."""
    value = 1.0 if normalization == "norm" or order == 0 else math.sqrt(2)
    if normalization == "integral":
        value /= math.sqrt(2 * math.pi)
    return value * (2 ** ((1 - order) / 2) if order else 1)


@lru_cache(maxsize=128)
def layout(orders, normalization):
    """Return signed real/imaginary selectors for flattened Cartesian axes."""
    indices, scales = [], []
    for order in orders:
        coefficient = scale(order, normalization)
        for component in range(2**order):
            phase = component.bit_count() % 4
            indices.append(2 * order - 1 + phase % 2 if order else 0)
            scales.append(coefficient if phase < 2 else -coefficient)
    return tuple(indices), tuple(scales)


def reference(module, vectors):
    """Apply the original circular recurrence and dense basis matrices."""
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    values = circular_harmonics.reference(vectors, max(module.orders, default=0))
    outputs = []
    for order in module.orders:
        value = (
            values[..., :1] if not order else values[..., 2 * order - 1 : 2 * order + 1]
        )
        coefficient = (
            1.0 if module.normalization == "norm" or order == 0 else math.sqrt(2)
        )
        if module.normalization == "integral":
            coefficient /= math.sqrt(2 * math.pi)
        outputs.append(value * coefficient)
    return module.to_cartesian(torch.cat(outputs, -1) if outputs else values[..., :0])


def evaluate(module, vectors):
    """Expand compact harmonics with signed column selection, avoiding GEMM."""
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_PLANAR_HARMONICS
        or not hasattr(module, "harmonic_indices")
    ):
        return reference(module, vectors)
    values = circular_harmonics.evaluate(
        vectors, max(module.orders, default=0), module.normalize
    )
    return values.index_select(-1, module.harmonic_indices) * module.harmonic_scales
