"""Fixed sparse coefficients for projected Cartesian delta/epsilon products."""

from .sparse_coefficients import bilinear_tables
from .torch.cartesian import normalized_coefficients


def coefficients(l1, l2, l3, expansions):
    """Fuse trace branches into compact sparse symmetric coordinates."""
    entries = []
    for trace, expansion in enumerate(expansions):
        rows = tuple(
            tuple((k, value) for k, value in enumerate(row) if value)
            for row in expansion.tolist()
        )
        for a, b, c, scale in normalized_coefficients(l1, l2, l3 - 2 * trace):
            entries.extend((a, b, k, scale * value) for k, value in rows[c])
    return bilinear_tables((3**l1, 3**l2, expansions[0].shape[-1]), tuple(entries))


def operands(module):
    return tuple(getattr(module, f"sparse_{k}") for k in range(9))
