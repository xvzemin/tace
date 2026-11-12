"""Batched sparse products for tensor-expression programs."""

from collections import defaultdict
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode


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
