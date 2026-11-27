"""Static work estimates for compiler-visible CUDA programs."""

from functools import lru_cache

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ...utils.metadata import decode_plans, parse_metadata
from .. import config


def wide_symmetric(metadata, channels):
    """Keep element-weighted CUDA contractions with wide term axes intact."""
    if not statically_known_true(channels >= config.SYMMETRIC_WIDE_CHANNELS):
        return False
    _, paths, _, _, rows = parse_metadata(metadata)
    return bool(paths) and len(rows) >= config.SYMMETRIC_WIDE_TERMS


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def ace_expansion_width(metadata):
    """Return the largest per-node channel table or intermediate product width."""
    dims, weighted, schedules = parse_metadata(metadata)
    if not weighted:
        return dims[-1]
    return max(
        (r[4] * r[5] for _, entries in schedules[3] for r, _ in entries), default=0
    )


def wide_ace(metadata, nodes):
    """Keep wide ACE contractions native when inlining can expand node-sized tables."""
    budget = config.ACE_INLINE_MAX_ELEMENTS
    if not budget:
        return False
    width = ace_expansion_width(metadata)
    return width > 256 and not statically_known_true(nodes * width <= budget)


def should_inline(enabled, estimate, *args):
    """Inline bounded programs during capture, without estimating eager work."""
    return (
        enabled
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
        and estimate(*args) <= config.COMPILE_MAX_INLINE_WORK
    )


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def frame_work(layout):
    """Count the retained angular rows, not their channel or batch copies."""
    return sum(len(rows) for *_, rows in parse_metadata(layout)[3])


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def alignment_work(key):
    """Count live geometry expressions after the requested transposes."""
    from ...utils.scalar import alignment_program

    nodes, outputs, _ = alignment_program(parse_metadata(key), "double")
    needed = {i for group in outputs for i in group}
    for i in reversed(range(len(nodes))):
        if i in needed and nodes[i][0] not in ("input", "constant"):
            needed.update(nodes[i][1:])
    return len(needed)


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def polynomial_work(kind, metadata, program):
    """Estimate expanded contraction work, including transposed factor roles."""
    from .polynomial import program_terms

    terms = program_terms(kind, program)
    description = parse_metadata(metadata) if kind != "indexed_plan" else None
    if kind in ("ace", "indexed_plan"):
        plans = decode_plans(metadata) if kind == "indexed_plan" else (description,)
        return sum(
            sum(len(entries) for _, entries in plans[plan][2][role])
            * (len(mapping) - 1)
            for plan, mapping, role, _, _ in terms
        )
    if kind == "linear":
        return len(description[4]) * len(terms)
    if kind in ("symmetric", "symmetric_matrix"):
        from .torch.symmetric import polynomial_batches

        return sum(
            len(signature) + 1
            for (_, _, signature), _ in polynomial_batches(metadata, program)
        )
    if kind == "channel":
        signatures = {
            tuple(i for i, index in enumerate(row[0]) if index >= 0)
            for row in description[2]
        }
        return sum(
            len(signature)
            for _, _, role, _, required in terms
            for signature in signatures
            if role in signature
            and all(
                i in signature
                for i in range(len(description[1]))
                if required & (1 << i)
            )
        )
    raise ValueError(f"Unknown polynomial kind {kind!r}.")


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def edge_work(metadata):
    """Estimate expression work without depending on batch sizes or tensors."""
    nodes, _ = parse_metadata(metadata)
    work = len(nodes)
    for operation, _, _, data in nodes:
        if operation == "frame":
            work += frame_work(data[0])
        elif operation == "product":
            from .torch.product import (
                BILINEAR_LITERAL_SIZE,
                bilinear_specification,
                product_groups,
            )

            specification = bilinear_specification(data[0])
            if specification is not None:
                # Large bilinear maps gather whole coefficient tables at once.
                count = len(specification[1])
                work += 2 * count if count <= BILINEAR_LITERAL_SIZE else 16
            else:
                work += sum(
                    len(signature) + 1 for signature, _ in product_groups(*data)
                )
        elif operation == "convolution":
            pending = [parse_metadata(data[1])]
            while pending:
                value = pending.pop()
                if isinstance(value, (tuple, list)):
                    pending.extend(value)
                else:
                    work += 1
    return work
