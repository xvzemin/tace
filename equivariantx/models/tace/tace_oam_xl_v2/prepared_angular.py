"""Reuse the fixed geometry part of sparse CG matrices within one step.

For each path, B[i,k] = sum_j CG[i,j,k] Y[j]. This changes the contraction
order only. Source-node features, radial weights, and parameter gradients
remain live. Angular-output adjoints retain the original sparse program.
"""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.pointwise import evaluate as pointwise
from equivariantx.utils.metadata import parse_metadata

GEOMETRY_SCOPE = ContextVar("oam_xl_prepared_angular", default=None)


@contextmanager
def geometry_scope():
    """Limit geometry reuse to one complete training invocation or capture."""
    if GEOMETRY_SCOPE.get() is not None:
        yield
        return
    token = GEOMETRY_SCOPE.set({})
    try:
        yield
    finally:
        GEOMETRY_SCOPE.reset(token)


@lru_cache(maxsize=64)
def angular_metadata(metadata):
    """Canonicalize equal and opposite linear angular forms across paths."""
    paths, width, weighted, *extra = parse_metadata(metadata)
    forms_by_group, lookups, mapped = {}, {}, []
    for path in paths:
        group = path[1]
        forms = forms_by_group.setdefault(group, [])
        lookup = lookups.setdefault(group, {})
        by_pair = defaultdict(list)
        for i, j, k, coefficient in path[-1]:
            by_pair[i, k].append((path[1] + j, coefficient))
        rows = []
        for (i, k), terms in by_pair.items():
            terms = sorted(terms)
            sign = -1.0 if terms[0][1] < 0 else 1.0
            canonical = tuple((j, coefficient * sign) for j, coefficient in terms)
            if canonical not in lookup:
                lookup[canonical] = len(forms)
                forms.append(canonical)
            rows.append((i, lookup[canonical], k, sign))
        mapped.append((path, tuple(rows)))
    offsets, forms = {}, []
    for group, values in sorted(forms_by_group.items()):
        offsets[group] = len(forms)
        forms.extend(values)
    mapped = tuple(
        (
            path[0],
            offsets[path[1]],
            *path[2:6],
            len(forms_by_group[path[1]]),
            *path[7:-1],
            rows,
        )
        for path, rows in mapped
    )
    return repr((mapped, width, weighted, *extra)), tuple(forms)


@lru_cache(maxsize=64)
def angular_tables(metadata, dtype, device):
    _, forms = angular_metadata(metadata)
    terms = max(map(len, forms))
    indices = torch.zeros((terms, len(forms)), dtype=torch.int32)
    coefficients = torch.zeros((terms, len(forms)), dtype=dtype)
    for column, form in enumerate(forms):
        for term, (index, coefficient) in enumerate(form):
            indices[term, column], coefficients[term, column] = index, coefficient
    return indices.to(device), coefficients.to(device)


def prepare_angular(metadata, value):
    """Build live angular forms, with optional invocation-scoped reuse."""
    cache = GEOMETRY_SCOPE.get()
    key = (metadata, value.data_ptr(), value.shape, value.stride(), value._version)
    if cache is not None and key in cache:
        return cache[key]
    indices, coefficients = angular_tables(metadata, value.dtype, value.device)
    out = value.new_empty((value.shape[0], indices.shape[1]))
    width = out.shape[1]
    pointwise(
        out,
        (value, indices, coefficients),
        f"""
        I row=i/{width}, column=i%{width}; T total=0;
        for(int term=0; term<{indices.shape[0]}; ++term) {{
            I at=term*{width}+column;
            total += p2[at]*p0[row*{value.stride(0)}+p1[at]*{value.stride(1)}];
        }}
        out[i]=total;
        """,
    )
    if cache is not None:
        # Retain the original to prevent its allocator address from being
        # reused while this invocation still holds a prepared value.
        cache[key] = out
        cache["input", key] = value
    return out


def angular_operands(metadata, program, operands):
    """Replace only angular input slots of a program with no angular output."""
    terms = parse_metadata(program)
    if any(role == 3 for _, _, pairs in terms for role, _ in pairs):
        raise ValueError("Angular-output programs must keep the original CG adjoint")
    values = list(operands)
    for slot in {mapping[3] for mapping, _, _ in terms}:
        values[slot] = prepare_angular(metadata, values[slot])
    return angular_metadata(metadata)[0], values
