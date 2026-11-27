"""Cached operator specifications."""

import json
import zlib
from ast import literal_eval
from dataclasses import dataclass
from functools import lru_cache
from operator import index

import torch

__all__ = ["KernelPlan", "compact_metadata", "kernel_plan", "parse_metadata"]


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def column_schedule(indices, size):
    """Identify contiguous channel broadcasts and their reduction adjoints."""
    if not indices:
        return None
    start, stop = min(indices), max(indices) + 1
    width = stop - start
    if start < 0 or stop > size or len(indices) % width:
        return None
    repeats = len(indices) // width
    if indices == tuple(range(start, stop)) * repeats:
        return start, width, repeats, False
    if indices == tuple(i for i in range(start, stop) for _ in range(repeats)):
        return start, width, repeats, True
    return None


def static_dimensions(tensor, dimensions=None):
    """Specialize model widths while leaving unselected batch dimensions dynamic."""
    dimensions = tuple(range(tensor.ndim)) if dimensions is None else dimensions
    if torch.compiler.is_compiling():
        for dimension in dimensions:
            torch._dynamo.mark_static(tensor, dimension)
    return tuple(index(tensor.shape[dimension]) for dimension in dimensions)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def parse_metadata(metadata: str):
    """Decode an immutable, tensor-free specification once at trace time."""
    if metadata.startswith("j:"):

        def immutable(value):
            return tuple(map(immutable, value)) if isinstance(value, list) else value

        return immutable(json.loads(zlib.decompress(bytes.fromhex(metadata[2:]))))
    if metadata.startswith("z:"):
        metadata = zlib.decompress(bytes.fromhex(metadata[2:])).decode()
    return literal_eval(metadata)


def numeric_rows(*description, rows_at=-1):
    """Encode a contraction specification with its rows in field ``rows_at``."""
    if len(description[rows_at]) < 4096:
        return compact_metadata(repr(description))
    # Decode large coefficient tables without building a Python syntax tree.
    payload = json.dumps(description, separators=(",", ":"), allow_nan=False).encode()
    return "j:" + zlib.compress(payload, level=1).hex()


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
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
    if len(metadata) < 4096 or metadata.startswith(("z:", "j:")):
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
@torch.compiler.assume_constant_result
def decode_plans(metadata):
    """Decode compressed contraction plans."""
    return tuple(
        parse_metadata(value)
        for value in parse_metadata(zlib.decompress(bytes.fromhex(metadata)).decode())
    )


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def independent_channel_groups(metadata):
    """Find disjoint channel groups for complementary inputs sharing one factor.

    The common factor's coordinate names the group. Every occupied coordinate
    of *every* role must belong to just one group, even in a transposed program.
    Keep the general evaluator for dense products, mixed widths and reductions.
    """
    width, dims, rows = parse_metadata(metadata)
    if (
        width not in (32, 64, 128)
        or len(dims) != 4
        or not 0 < len(rows) <= 128
        or any(not 0 < d <= len(rows) for d in dims)
    ):
        return None
    missing = {tuple(k for k, i in enumerate(indices) if i < 0) for indices, _ in rows}
    if len(missing) != 2 or any(len(mask) != 1 or mask[0] == 3 for mask in missing):
        return None
    (pivot,) = set(range(3)) - {k for (k,) in missing}
    groups = [[] for _ in range(dims[pivot])]
    owner = {}
    for indices, coefficient in rows:
        group = indices[pivot]
        for role, i in enumerate(indices):
            if i >= dims[role]:
                return None
            if i >= 0 and owner.setdefault((role, i), group) != group:
                return None
        groups[group].append((indices, coefficient))
    if len(owner) != sum(dims) or max(map(len, groups)) > 8:
        return None
    return tuple(tuple(group) for group in groups)


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
