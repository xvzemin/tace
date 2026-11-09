"""Cached operator specifications."""

import zlib
from ast import literal_eval
from dataclasses import dataclass
from functools import lru_cache

import torch

__all__ = ["KernelPlan", "compact_metadata", "kernel_plan", "parse_metadata"]


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def parse_metadata(metadata: str):
    """Decode an immutable, tensor-free specification once at trace time."""
    if metadata.startswith("z:"):
        metadata = zlib.decompress(bytes.fromhex(metadata[2:])).decode()
    return literal_eval(metadata)


@lru_cache(maxsize=256)
def compact_metadata(metadata: str):
    """Compact a literal operator specification.

    Parameters
    ----------
    metadata : str
        Tensor-free specification accepted by :func:`parse_metadata`.

    Returns
    -------
    str
        Compressed specification, or the input if short or already compressed.

    Examples
    --------
    >>> spec = repr(tuple(range(2000)))
    >>> parse_metadata(compact_metadata(spec)) == parse_metadata(spec)
    True
    """
    if len(metadata) < 4096 or metadata.startswith("z:"):
        return metadata
    return "z:" + zlib.compress(metadata.encode()).hex()


@dataclass(eq=False)
class KernelPlan:
    """Tensor-free scheduling metadata shared by compiled calls."""

    path_data: tuple
    weight_numel: int
    has_unweighted: bool
    sparse_paths: tuple
    scalar_paths: tuple = ()
    harmonic_degrees: tuple = ()


@lru_cache(maxsize=256)
def kernel_plan(metadata):
    return KernelPlan(*parse_metadata(metadata))


def encode_plans(plans):
    """Encode a collection of contraction plans for exported operators."""
    return zlib.compress(repr(tuple(plans)).encode()).hex()


@lru_cache(maxsize=128)
def decode_plans(metadata):
    """Decode compressed contraction plans."""
    return tuple(
        parse_metadata(value)
        for value in parse_metadata(zlib.decompress(bytes.fromhex(metadata)).decode())
    )


def channel_product_layout(metadata):
    """Return flat dimensions and sparse channel-strided contractions."""
    width, dims, rows = parse_metadata(metadata)
    if width:
        return tuple(d * width for d in dims), tuple(
            (
                tuple(i * width if i >= 0 else -1 for i in indices),
                coefficient,
                width,
                (1,) * len(dims),
            )
            for indices, coefficient in rows
        )
    return dims, rows
