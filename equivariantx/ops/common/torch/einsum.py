"""Contraction paths that preserve symbolic batch dimensions."""

from functools import lru_cache

import torch

SYMBOLIC_PATH_SIZE = 128


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def contraction_path(equation, shapes):
    """Estimate symbolic extents for path cost without constraining execution."""
    try:
        import opt_einsum
    except ImportError:
        return None

    tokens, _, _ = opt_einsum.parser.parse_einsum_input(
        (equation, *shapes), shapes=True
    )
    sizes = {}
    for token, shape in zip(tokens.split(","), shapes):
        for label, size in zip(token, shape):
            if size > 1:
                sizes[label] = size
    planned = tuple(
        tuple(
            sizes.get(label, SYMBOLIC_PATH_SIZE) if size < 0 else size
            for label, size in zip(token, shape)
        )
        for token, shape in zip(tokens.split(","), shapes)
    )
    path, _ = opt_einsum.contract_path(equation, *planned, shapes=True, optimize="auto")
    return [index for pair in path for index in pair]


@torch.compiler.allow_in_graph
def einsum(equation, *operands):
    """Keep eager path selection and plan symbolic contractions from model widths."""
    if len(operands) < 3 or not any(
        isinstance(size, torch.SymInt) for value in operands for size in value.shape
    ):
        return torch.einsum(equation, *operands)
    shapes = tuple(
        tuple(-1 if isinstance(size, torch.SymInt) else size for size in value.shape)
        for value in operands
    )
    return torch.ops.aten.einsum.default(
        equation, list(operands), path=contraction_path(equation, shapes)
    )
