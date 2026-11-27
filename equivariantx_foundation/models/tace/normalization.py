"""Opt-in provider for a TACE interaction's existing spherical RMS normalization."""

from functools import partial
from struct import Struct

import torch
from e3nn import o3

from equivariantx.nn import MergeRmsNorm
from tace.utils.env import acceleration_enabled

from .interactions import _SPHERICAL_TYPES
from .matching import custom_forward


def _standard_weights(norm):
    default = norm.default_block_weights
    if norm.block_weights == default:
        return True
    # Released checkpoints can store the same analytic weights as FP32.
    # Match that exact serialization; never round or replace the live weights.
    floats = Struct(f"{len(default)}f")
    return norm.block_weights == floats.unpack(floats.pack(*default))


def _eligible(norm):
    # Only the multi-block, affine, spherical FP32 case. One scalar block can
    # take the same torch RMS fast path on both backends. Other layouts,
    # unweighted norms and other norm classes have not been benchmarked here.
    return (
        type(norm) is MergeRmsNorm
        and type(norm.irreps) is o3.Irreps
        and norm.layout == "mul_ir"
        and 1 < len(norm.widths) <= 16
        and norm.p == norm.max_rms == 1
        and not norm.scalar_blocks
        and _standard_weights(norm)
        and norm.weight is not None
        and norm.weight.is_cuda
        and norm.weight.dtype == torch.float32
        and norm.bias is None
    )


def _use_backend(norm, *, backend):
    # Keep the same forward, affine gain, state schema and runtime guards.
    norm.backend = backend
    return norm


def normalization_regions(rep, *, backend, automatic):
    for i, layer in enumerate(rep.interactions):
        if type(layer) not in _SPHERICAL_TYPES:
            continue
        for name in ("norm1", "norm2"):
            norm = getattr(layer, name, None)
            if type(norm) is not MergeRmsNorm:
                continue
            path = f"interactions.{i}.{name}"
            old = getattr(norm, "_foundation_regional_provider", None)
            if old is not None:
                if old[0] != norm.backend:
                    raise ValueError(
                        f"Cannot retarget {path}: its provider was changed outside the planner."
                    )
                if automatic or old[0] == backend:
                    yield path, old[0], None, old[1]
                    continue
            elif automatic or backend != "cuda" or norm.backend != "torch":
                continue
            # The CUDA branch bypasses inverse_rms; even an instance-level
            # override of this helper (not just of forward) must stay native.
            if (
                acceleration_enabled("eqx")
                or custom_forward(layer)
                or custom_forward(norm, recursive=True)
                or "inverse_rms" in vars(norm)
                or (backend == "cuda" and not _eligible(norm))
            ):
                if old is not None:
                    raise ValueError(
                        f"Cannot retarget {path}: customized or unsupported normalization. "
                        "Use the original host model."
                    )
                continue
            yield (
                path,
                backend,
                partial(_use_backend, backend=backend),
                f"spherical multi-block MergeRmsNorm using {backend}",
            )
