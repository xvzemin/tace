"""Batched sparse products for tensor-expression programs."""

from collections import defaultdict
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ..sparse_coefficients import bilinear_tables
from .literal_bilinear import evaluate as literal_bilinear
from .sparse_bilinear import evaluate as sparse_bilinear

BILINEAR_LITERAL_SIZE = 256


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def bilinear_specification(metadata):
    """Recognize complete real bilinear terms without absent factor roles."""
    _, dimensions, entries = metadata
    if (
        len(dimensions) != 3
        or not entries
        or any(
            len(indices) != 3 or any(index < 0 for index in indices)
            for indices, _ in entries
        )
    ):
        return None
    return dimensions, tuple(
        (*indices, coefficient) for indices, coefficient in entries
    )


@lru_cache(maxsize=128)
def bilinear_constants(specification, device, dtype):
    return bilinear_tables(*specification, dtype=dtype, device=device)


def bilinear_product(metadata, role, operands, rows):
    """Use exact transposes instead of expanded-term scatter adjoints."""
    specification = bilinear_specification(metadata)
    if specification is None:
        return None
    compiling = torch.compiler.is_compiling() or get_proxy_mode() is not None
    dimensions, entries = specification
    use_literal = compiling and len(entries) <= BILINEAR_LITERAL_SIZE
    if not operands[0].is_cuda and not use_literal:
        return None
    channels = metadata[0]
    values = [
        value.reshape(value.shape[0], dimensions[i], channels).transpose(1, 2)
        for i, value in enumerate(operands)
        if i != role
    ]
    if use_literal:
        result = literal_bilinear(*values, dimensions, entries, role)
    else:
        tracing = is_fake(values[0]) or torch._C._are_functorch_transforms_active()
        tables = (
            bilinear_constants.__wrapped__
            if compiling or tracing
            else bilinear_constants
        )
        result = sparse_bilinear(
            *values, tables(specification, values[0].device, values[0].dtype), role
        )
    return (
        result.transpose(1, 2)
        .expand(rows, dimensions[role], channels)
        .reshape(rows, dimensions[role] * channels)
    )


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def product_groups(metadata, role, required):
    """Group terms with the same occupied factor roles."""
    groups = defaultdict(list)
    for indices, coefficient in metadata[2]:
        if indices[role] < 0 or any(
            index < 0 for k, index in enumerate(indices) if required & (1 << k)
        ):
            continue
        signature = tuple(
            k for k, index in enumerate(indices) if k != role and index >= 0
        )
        groups[signature].append(
            (indices[role], tuple(indices[k] for k in signature), coefficient)
        )
    return tuple((signature, tuple(terms)) for signature, terms in groups.items())


@lru_cache(maxsize=256)
def product_indices(metadata, role, required, device, dtype):
    """Create immutable angular indices and coefficients."""
    return tuple(
        (
            torch.tensor([term[0] for term in terms], dtype=torch.long, device=device),
            tuple(
                torch.tensor(
                    [term[1][i] for term in terms], dtype=torch.long, device=device
                )
                for i in range(len(signature))
            ),
            torch.tensor([term[2] for term in terms], dtype=dtype, device=device),
        )
        for signature, terms in product_groups(metadata, role, required)
    )


def product(metadata, role, required, operands, rows):
    """Evaluate all sparse angular terms with differentiable gather/reduce ops."""
    if not config.TORCH_REFERENCE and config.TORCH_BILINEAR_PRODUCT:
        result = bilinear_product(metadata, role, operands, rows)
        if result is not None:
            return result
    if (
        config.TORCH_PRODUCT_MAX_TERMS
        and len(metadata[2]) <= config.TORCH_PRODUCT_MAX_TERMS
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return unrolled_product(metadata, role, required, operands, rows)
    width, dims, _ = metadata
    tracing = (
        torch.compiler.is_compiling()
        or is_fake(operands[0])
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    )
    indices = product_indices.__wrapped__ if tracing else product_indices
    indices = indices(metadata, role, required, operands[0].device, operands[0].dtype)
    output = operands[0].new_zeros((rows, dims[role], width))
    for (signature, _), (destination, columns, coefficient) in zip(
        product_groups(metadata, role, required), indices
    ):
        term = coefficient[None, :, None]
        for k, column in zip(signature, columns):
            value = operands[k].reshape(operands[k].shape[0], dims[k], width)
            term = term * value.index_select(1, column)
        output = output.index_add(
            1, destination, term.expand(rows, destination.numel(), width)
        )
    return output.flatten(1)


def unrolled_product(metadata, role, required, operands, rows):
    """Expose small angular sums to the compiler without index reductions."""
    width, dims, _ = metadata
    parts = [[] for _ in range(dims[role])]
    factors = {}
    for signature, terms in product_groups(metadata, role, required):
        for output, columns, coefficient in terms:
            term = None
            for k, column in zip(signature, columns):
                key = k, column
                if key not in factors:
                    value = operands[k].reshape(operands[k].shape[0], dims[k], width)
                    factors[key] = value[:, column, :]
                value = factors[key]
                term = value if term is None else term * value
            term = (
                operands[0].new_full((rows, width), coefficient)
                if term is None
                else term * coefficient
            )
            parts[output].append(term)
    values = [
        sum(terms[1:], terms[0]).expand(rows, width)
        if terms
        else operands[0].new_zeros((rows, width))
        for terms in parts
    ]
    return (
        torch.stack(values, dim=1).flatten(1)
        if values
        else operands[0].new_empty((rows, 0))
    )
