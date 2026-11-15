"""Balanced sums with ordinary PyTorch differentiation."""


def sum_pairwise(values):
    """Sum a nonempty sequence using a balanced addition tree."""
    values = list(values)
    while len(values) > 1:
        values = [
            values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)
        ] + values[len(values) // 2 * 2 :]
    return values[0]


def add_pairwise(partials, value):
    """Append one term to an in-place list of binary-tree partial sums."""
    for i, previous in enumerate(partials):
        if previous is None:
            partials[i] = value
            return
        value = previous + value
        partials[i] = None
    partials.append(value)
