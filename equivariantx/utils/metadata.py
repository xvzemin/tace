"""Cached operator specifications."""

from ast import literal_eval
from functools import lru_cache

import torch

__all__ = ["parse_metadata"]


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def parse_metadata(metadata: str):
    """Decode an immutable, tensor-free specification once at trace time."""
    return literal_eval(metadata)
