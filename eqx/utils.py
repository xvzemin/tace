"""Shared utilities for operator specifications."""

from ast import literal_eval
from functools import lru_cache

import torch

__all__ = ["parse_metadata"]


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def parse_metadata(metadata: str):
    """Decode a tensor-free specification shared by operators and CUDA plans.

    Parameters
    ----------
    metadata : str
        Literal representation of an immutable, tensor-free specification.

    Returns
    -------
    object
        Decoded specification. Repeated calls reuse a cache of up to 256 entries.

    Notes
    -----
    Specifications must not contain live model objects. During compilation,
    decoding runs at trace time and does not enter the tensor graph.
    """
    return literal_eval(metadata)
