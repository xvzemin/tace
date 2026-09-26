"""Degree-major and order-major Wigner matrix indexing."""

import math


def wigner_orders(
    global_dim: int, local_dim: int, *, lmax: int = 0, mmax: int = 0
) -> tuple[int, int]:
    """Infer retained degrees and orders, checking the requested coverage."""
    degree = int(math.sqrt(global_dim)) - 1
    if (degree + 1) ** 2 != global_dim or degree < lmax:
        raise ValueError("Wigner global dimension must cover every O(3) degree.")
    missing = global_dim - local_dim
    if missing < 0:
        raise ValueError("Wigner has an incompatible local dimension.")
    # Removing the highest local orders removes n * (n + 1) rows.
    omitted = (int(math.sqrt(4 * missing + 1)) - 1) // 2
    order = degree - omitted
    if omitted * (omitted + 1) != missing or not mmax <= order <= degree:
        raise ValueError("Wigner local dimension must cover all required orders.")
    return degree, order


def wigner_indices(degree: int, mmax: int, lmax: int) -> list[int]:
    """Return rows for orders 0, +1, -1, ... of one degree block."""
    rows = [degree]
    for m in range(1, min(degree, mmax) + 1):
        rows.extend(
            (
                degree + (2 * m - 1) * (lmax + 1) - m * m,
                degree + 2 * m * (lmax + 1) - m * (m + 1),
            )
        )
    return rows
