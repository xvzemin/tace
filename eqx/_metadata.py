"""Cached decoding of immutable operator specifications."""

from ast import literal_eval
from functools import lru_cache


@lru_cache(maxsize=256)
def parse_metadata(metadata: str):
    """Decode a tensor-free specification shared by operators and CUDA plans.

    The returned tuples are immutable and must not contain live model objects.
    Keeping this cache independent of any operator avoids importing a backend
    merely to read its metadata.
    """
    return literal_eval(metadata)
